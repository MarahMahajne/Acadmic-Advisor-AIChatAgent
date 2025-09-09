# ingestion/majors_extractor.py
"""
MajorsExtractor — fast single-call (program-only) with English names
--------------------------------------------------------------------
• Reads full text via PDFProcessor (fallback: pdfplumber)
• Single Gemini call (strict JSON, braces escaped) → fast
• Filters to true program/degree titles (≈15), merges dupes
• Adds english_name via translator (supports both MajorsTranslator or translate_batch)
• Writes extracted_majors.json in the schema HybridRAG expects
"""

from __future__ import annotations
import os, re, io, json
from typing import List, Dict, Any, Optional
from functools import lru_cache

from dotenv import load_dotenv

# ---- PDF text ----
try:
    # same-dir import style for your tree: ingestion/PDFProcessor.py
    from .PDFProcessor import PDFProcessor  # type: ignore
except Exception:
    PDFProcessor = None  # type: ignore

# ---- translator (support both styles) ----
try:
    # style A: class MajorsTranslator with translate_name_map(list[str]) -> dict
    from .translator import MajorsTranslator  # type: ignore
except Exception:
    MajorsTranslator = None  # type: ignore

try:
    # style B: module-level translate_batch(list[str], target_lang="en") -> list[str]
    from . import translator as _translator_mod  # type: ignore
except Exception:
    _translator_mod = None  # type: ignore

# ---- Gemini / LangChain ----
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    StrOutputParser = None

DEBUG = os.getenv("MAJORS_DEBUG", "0") == "1"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ========================= Debug helpers =========================
def _dbg(*a: Any) -> None:
    if DEBUG:
        print("[majors_extractor:debug]", *a, flush=True)

def _debug_blob(tag: str, text: str, n: int = 350) -> None:
    if DEBUG:
        s = (text or "")[:n].replace("\n", " ")
        print(f"[debug] {tag}_len={len(text or '')}")
        print(f"[debug] {tag}_head: {s}{'...' if len(text or '')>n else ''}")


# ========================= Utilities =========================
@lru_cache(maxsize=4096)
def _has_hebrew(text: str) -> bool:
    if not text:
        return False
    return any("\u0590" <= ch <= "\u05FF" for ch in text)

_CODEBLOCK_START = re.compile(r"^\s*```(?:json)?", re.IGNORECASE)
_CODEBLOCK_END   = re.compile(r"\s*```\s*$")

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = _CODEBLOCK_START.sub("", s)
    s = _CODEBLOCK_END.sub("", s)
    return s.strip()

def _safe_json_loads(s: str, default: Any = None) -> Any:
    if not s:
        return default
    s1 = _strip_code_fences(s)
    # try direct
    try:
        return json.loads(s1)
    except Exception:
        pass
    # salvage inner {...} or [...]
    m = re.search(r"(\{.*\}|\[.*\])", s1, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # minor repairs: trailing commas, single quotes
    s2 = re.sub(r",\s*([}\]])", r"\1", s1)
    s2 = s2.replace("'", '"')
    try:
        return json.loads(s2)
    except Exception:
        return default


# ---------- program/title heuristics (to keep ~true majors only) ----------
_PROGRAM_TOKENS = [
    "תואר", "דו-חוגי", "חד-חוגי", "B.A", "B.Sc", "BSc", "B.Ed", "BSN", "M.A", "M.Sc", "MSc",
    " :ב",  # common pattern like '... :בניהול ...'
]
_GENERIC_BAD_PREFIXES = [
    "ניהול", "כלכלה", "סטטיסטיקה", "פסיכולוגיה", "סוציולוגיה", "תכנות", "אלגוריתמים",
    "קבלת החלטות", "עקרונות השיווק", "התנהגות ארגונית",
]

def _is_program_name(name: str) -> bool:
    if not name:
        return False
    s = name.strip()
    if s.startswith("תואר ראשון") or s.startswith("תואר שני"):
        return True
    if any(tok in s for tok in _PROGRAM_TOKENS):
        return True
    if any(x in s for x in ["B.A", "B.Sc", "M.A", "M.Sc", "BSN"]):
        return True
    return False

def _normalize_key(name: str) -> str:
    s = re.sub(r"\s+", " ", name or "").strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[:：]+$", "", s)  # drop trailing colons
    return s

def _merge_program_rows(rows: List[Dict[str, Any]], source_path: str) -> List[Dict[str, Any]]:
    """Keep only true programs; merge by normalized name; keep richest courses/keywords."""
    filtered = [r for r in rows if _is_program_name(r.get("original_name",""))]
    merged: Dict[str, Dict[str, Any]] = {}
    for r in filtered:
        key = _normalize_key(r.get("original_name",""))
        if not key:
            continue
        cur = merged.get(key)
        if not cur:
            merged[key] = {
                "original_name": key,
                "english_name": r.get("english_name",""),
                "keywords": list(dict.fromkeys((r.get("keywords") or []))),
                "sample_courses": list(dict.fromkeys((r.get("sample_courses") or []))),
                "source": source_path,
            }
        else:
            en = r.get("english_name","")
            if en and len(en) > len(cur.get("english_name","")):
                cur["english_name"] = en
            cur["keywords"] = list(dict.fromkeys(cur["keywords"] + (r.get("keywords") or [])))
            cur["sample_courses"] = list(dict.fromkeys(cur["sample_courses"] + (r.get("sample_courses") or [])))
    # prune very short names unless they contain degree tokens
    out = []
    for k, r in merged.items():
        if len(k) < 8 and not any(tok in k for tok in _PROGRAM_TOKENS):
            continue
        out.append(r)
    return out


# ========================= PDF Text =========================
def _extract_text(pdf_path: str) -> str:
    """Read the entire PDF into a single text blob (prefer PDFProcessor)."""
    if PDFProcessor is not None:
        try:
            with open(pdf_path, "rb") as f:
                data = io.BytesIO(f.read())
            txt = PDFProcessor.extract_text_from_pdf(data)
            if txt and txt.strip():
                return txt
        except Exception as e:
            _dbg("PDFProcessor failed:", e)
    # fallback: pdfplumber
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return ""
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                chunks.append(f"[PAGE {i}]\n{t}\n")
    except Exception as e:
        _dbg("pdfplumber failed:", e)
    return "\n".join(chunks)


# ========================= LLM Extraction (single call) =========================
EXTRACT_SYSTEM = (
    "You are a careful academic catalog extractor. Return ONLY valid JSON. "
    "Preserve names and course titles exactly as in the source (do NOT translate)."
)

# IMPORTANT: literal braces are escaped {{ }}
EXTRACT_USER_TMPL = (
    "From the text, extract an array of **degree/program titles** (majors) and their representative sample courses.\n"
    "Return ONLY a JSON array where each item is:\n"
    "{{\n"
    "  \"original_name\": string,          # exact degree/program title as printed (Hebrew/native if Hebrew)\n"
    "  \"sample_courses\": [string],       # 3–20 course titles from that program section (verbatim)\n"
    "  \"keywords\": [string]              # 3–10 topical tags if inferable (optional)\n"
    "}}\n\n"
    "CRITICAL rules:\n"
    "- Extract ONLY program-level titles (e.g., lines containing degree markers like 'תואר', 'דו-חוגי', 'B.A.', 'B.Sc.', 'M.A.', 'BSN').\n"
    "- Do NOT output generic topic words as separate items (e.g., 'ניהול', 'כלכלה', 'סטטיסטיקה').\n"
    "- Group courses under the correct program; keep wording exactly as written.\n"
    "- JSON only. No code fences, no commentary.\n\n"
    "TEXT:\n---\n{full_text}\n---\nJSON:"
)

def _make_llm(model_env: str = "GEMINI_MODEL", default_model: str = DEFAULT_MODEL, temperature: float = 0.0):
    model_name = os.getenv(model_env, default_model)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def _extract_raw_items_single_call(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text.strip():
        return []
    if not (ChatGoogleGenerativeAI and ChatPromptTemplate and StrOutputParser):
        _dbg("langchain_google_genai not installed")
        return []

    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_SYSTEM),
        ("human",  EXTRACT_USER_TMPL),
    ])
    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"full_text": raw_text})
    _debug_blob("llm_raw", out)

    data = _safe_json_loads(out, default=[])
    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        seen = set()
        for obj in data:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("original_name", "")).strip()
            if not name or name in seen:
                continue
            # keep as given (no translation here)
            courses = obj.get("sample_courses") or []
            if not isinstance(courses, list):
                courses = [str(x).strip() for x in str(courses).split(",")]
            courses = [str(c).strip() for c in courses if str(c).strip()][:20]
            keywords = obj.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = [str(x).strip() for x in str(keywords).split(",")]
            keywords = [k for k in (str(x).strip() for x in keywords) if k][:10]
            items.append({
                "original_name": name,
                "sample_courses": courses,
                "keywords": keywords,
            })
            seen.add(name)
    return items


# ========================= Translation layer =========================
def _translate_names(names: List[str]) -> Dict[str, str]:
    """
    Try MajorsTranslator first; fall back to translate_batch; else identity.
    """
    # A) class MajorsTranslator
    if MajorsTranslator is not None:
        try:
            tr = MajorsTranslator()
            mapping = tr.translate_name_map(names)  # expected API from your note
            if isinstance(mapping, dict):
                return {k: (v or k) for k, v in mapping.items()}
        except Exception as e:
            _dbg("MajorsTranslator failed:", e)

    # B) function translate_batch
    if _translator_mod is not None:
        tx = getattr(_translator_mod, "translate_batch", None)
        if callable(tx):
            try:
                res = tx(names, target_lang="en")
                if isinstance(res, list) and len(res) == len(names):
                    return {names[i]: (res[i] or names[i]) for i in range(len(names))}
            except Exception as e:
                _dbg("translate_batch failed:", e)

    # C) identity fallback
    return {n: n for n in names}


# ========================= Public API =========================
def build_majors_profiles(pdf_path: str, out_json: str = "extracted_majors.json") -> str:
    """
    Extract majors (program titles) + courses (no translation), add english_name via translator,
    filter to real programs, merge duplicates, and save JSON.
    """
    load_dotenv()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = _extract_text(pdf_path)
    _debug_blob("raw_text", full_text)

    items = _extract_raw_items_single_call(full_text)

    # normalize early to expected shape
    prelim = []
    for it in items:
        name = str(it.get("original_name", "")).strip()
        if not name:
            continue
        prelim.append({
            "original_name": name,
            "english_name": "",  # filled later
            "keywords": it.get("keywords", []) or [],
            "sample_courses": it.get("sample_courses", []) or [],
            "source": os.path.basename(pdf_path),
        })

    # keep only real programs; merge dupes
    merged = _merge_program_rows(prelim, source_path=os.path.basename(pdf_path))

    # translate names → english_name
    names = [r["original_name"] for r in merged]
    name_map = _translate_names(names)
    for r in merged:
        en = str(name_map.get(r["original_name"], r["original_name"])).strip()
        if _has_hebrew(en):
            en = r["original_name"]
        r["english_name"] = en

    # write
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[majors_extractor] Wrote {len(merged)} majors -> {out_json}")
    return out_json
