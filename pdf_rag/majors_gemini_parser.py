# pdf_rag/majors_gemini_parser.py
"""
Gemini-powered majors extractor + built-in English translation.
Outputs **majors_profiles.json** with this exact schema per item:
{
  "original_name": str,          # exact as in PDF (Hebrew/native)
  "english_name": str,           # translated/conventional English name
  "keywords": [str],
  "sample_courses": [str],       # preserved wording from PDF (no translation)
  "source": str
}

Usage:
    from pdf_rag.majors_gemini_parser import build_major_profiles
    build_major_profiles("data/majors.pdf", out_json="majors_profiles.json")

Env:
    GOOGLE_API_KEY (required)
    GEMINI_MODEL  (default: gemini-2.0-flash)
"""
from __future__ import annotations

import os
import re
import io
import json
from typing import List, Dict, Any
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Local PDF text extractor (already in your repo)
from PDFProcessor import PDFProcessor

load_dotenv()

# ============================ Helpers ============================

_CODEBLOCK_START = re.compile(r"^```(?:json)?", re.IGNORECASE)
_CODEBLOCK_END = re.compile(r"```\s*$")


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
    s = _strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        # try to salvage inner JSON
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # minimal repairs: trailing commas & single quotes
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        s2 = s2.replace("'", '"')
        try:
            return json.loads(s2)
        except Exception:
            return default


def _make_llm(model_env: str = "GEMINI_MODEL", default_model: str = "gemini-2.0-flash", temperature: float = 0.2):
    model_name = os.getenv(model_env, default_model)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


@lru_cache(maxsize=4096)
def _has_hebrew(text: str) -> bool:
    if not text:
        return False
    return any("\\u0590" <= ch <= "\\u05FF" for ch in text)


# ============================ Extraction ============================

EXTRACT_SYSTEM = (
    "You are a careful academic catalog extractor. Return ONLY valid JSON.\\n"
    "Preserve names and course titles exactly as in the source (do NOT translate)."
)

# IMPORTANT: All literal braces in the JSON example are escaped as {{ and }}
EXTRACT_USER_TMPL = (
    "From the text, extract an array of majors and their representative sample courses.\n"
    "Return ONLY a JSON array where each item is:\n"
    "{{\n"
    "  \"original_name\": string,          # exact major/program name as it appears (keep Hebrew if Hebrew)\n"
    "  \"sample_courses\": [string],       # 3–20 short course titles (preserve wording, do NOT translate)\n"
    "  \"keywords\": [string]              # 3–10 short topical tags if inferable (optional)\n"
    "}}\n\n"
    "Rules:\n"
    "- No commentary, no code fences. JSON only.\n"
    "- If a field is missing, omit it.\n"
    "- Group courses under the correct major.\n\n"
    "TEXT:\n---\n{chunk}\n---\nJSON:"
)

_extract_prompt = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_SYSTEM),
    ("human", EXTRACT_USER_TMPL),
])
_extract_parser = StrOutputParser()


def _extract_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        data = io.BytesIO(f.read())
    return PDFProcessor.extract_text_from_pdf(data)


def _extract_raw_items(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text.strip():
        return []
    llm = _make_llm()
    chain = _extract_prompt | llm | _extract_parser
    out = chain.invoke({"chunk": raw_text})
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


# ============================ Translation ============================

TRANSLATE_SYSTEM = (
    "You translate academic program names into conventional English major names.\\n"
    "Return ONLY JSON mapping original->english. No commentary."
)

TRANSLATE_USER_TMPL = (
    "Given this JSON array of original (native/Hebrew) major names, return a JSON object mapping\\n"
    "each original name to a clean English major name. Keep conventional terms (e.g., 'Computer Science',\\n"
    "'Electrical Engineering', 'Industrial Engineering and Management').\\n\\n"
    "Names JSON array:\\n{names}\\n\\nJSON mapping only:"
)

_translate_prompt = ChatPromptTemplate.from_messages([
    ("system", TRANSLATE_SYSTEM),
    ("human", TRANSLATE_USER_TMPL),
])
_translate_parser = StrOutputParser()


def _translate_names_to_english(names: List[str]) -> Dict[str, str]:
    if not names:
        return {}
    llm = _make_llm()
    payload = json.dumps(sorted(list({n for n in names if n})), ensure_ascii=False)
    chain = _translate_prompt | llm | _translate_parser
    out = chain.invoke({"names": payload})
    mapping = _safe_json_loads(out, default={})
    if not isinstance(mapping, dict):
        return {n: n for n in names}
    # Ensure all names mapped
    for n in names:
        if n not in mapping or not str(mapping[n]).strip():
            mapping[n] = n
    return mapping


# ============================ Public API ============================

def build_major_profiles(pdf_path: str, out_json: str = "majors_profiles.json") -> str:
    """Extract majors from the PDF, translate names to English, and save JSON.
    Output matches what `HybridRAG.load_majors_from_json` expects.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) Extract full text and parse raw items
    raw_text = _extract_text(pdf_path)
    items = _extract_raw_items(raw_text)

    # 2) Build unique name list and translate to English in ONE batch
    unique_names = [it["original_name"] for it in items]
    name_map = _translate_names_to_english(unique_names)

    # 3) Normalize payload to final schema
    payload: List[Dict[str, Any]] = []
    for it in items:
        on = it["original_name"].strip()
        en = str(name_map.get(on, on)).strip()
        # If English is clearly Hebrew (LLM failed), fallback to original
        if _has_hebrew(en):
            en = on
        payload.append({
            "original_name": on,
            "english_name": en,
            "keywords": it.get("keywords", []),
            "sample_courses": it.get("sample_courses", []),
            "source": os.path.basename(pdf_path),
        })

    # 4) Persist
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[majors_gemini_parser] Wrote {len(payload)} majors -> {out_json}")
    return out_json
