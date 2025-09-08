# pdf_rag/majors_extractor.py
"""
MajorsExtractor 

What this file does
-------------------
• Reads the majors PDF (using the PDFProcessor)
• Uses Gemini to extract majors in the ORIGINAL language (Hebrew/native)
• Keeps `sample_courses` + `keywords` exactly as written (no translation here)
• Calls the separate translator module to add `english_name`
• Writes `extracted_majors.json` in the schema expected by HybridRAG.load_majors_from_json()

Item schema produced here (one per major):
{
  "original_name": str,          # as in PDF (Hebrew/native)
  "english_name": str,           # added by translator; conventional English name
  "keywords": [str],             # short topical tags (optional)
  "sample_courses": [str],       # preserved; not translated
  "source": str                  # the PDF file path/basename
}

Environment variables
---------------------
- GOOGLE_API_KEY  (required)
- GEMINI_MODEL    (default: "gemini-2.0-flash")

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
from PDFProcessor import PDFProcessor
from .translator import MajorsTranslator

load_dotenv()

# ----------------------------- Helpers -----------------------------

_CODEBLOCK_START = re.compile(r"^```(?:json)?", re.IGNORECASE)
_CODEBLOCK_END = re.compile(r"```\\s*$")

def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences that LLMs sometimes add around JSON."""
    if not s:
        return s
    s = s.strip()
    s = _CODEBLOCK_START.sub("", s)
    s = _CODEBLOCK_END.sub("", s)
    return s.strip()

def _safe_json_loads(s: str, default: Any = None) -> Any:
    """
    Best-effort JSON loader that:
    - strips code fences,
    - tries to salvage the innermost {..} or [..],
    - fixes trailing commas and single quotes.
    """
    if not s:
        return default
    s = _strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        # salvage innermost JSON
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # minimal repairs
        s2 = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
        s2 = s2.replace("'", '"')              # single -> double quotes
        try:
            return json.loads(s2)
        except Exception:
            return default

def _make_llm(model_env: str = "GEMINI_MODEL", default_model: str = "gemini-2.0-flash", temperature: float = 0.2):
    """Build a Gemini chat model instance with safe defaults."""
    model_name = os.getenv(model_env, default_model)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

@lru_cache(maxsize=4096)
def _has_hebrew(text: str) -> bool:
    """Heuristic: detect if a string contains any Hebrew codepoints (U+0590..U+05FF)."""
    if not text:
        return False
    return any("\u0590" <= ch <= "\u05FF" for ch in text)

# --------------------------- Extraction LLM --------------------------

EXTRACT_SYSTEM = (
    "You are a careful academic catalog extractor. Return ONLY valid JSON.\n"
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

# ------------------------------ Core ------------------------------

def _extract_text(pdf_path: str) -> str:
    """Read the entire PDF into a single text blob using PDFProcessor."""
    with open(pdf_path, "rb") as f:
        data = io.BytesIO(f.read())
    return PDFProcessor.extract_text_from_pdf(data)

def _extract_raw_items(raw_text: str) -> List[Dict[str, Any]]:
    """
    Run Gemini on the full PDF text and parse a clean list of items.
    Each item includes: original_name, sample_courses, keywords (no translation).
    """
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

# --------------------------- Public API ---------------------------

def build_major_profiles(pdf_path: str, out_json: str = "extracted_majors.json") -> str:
    """
    Extract majors (original language), translate names to English via translator,
    and save JSON. Output matches what `HybridRAG.load_majors_from_json` expects.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) Extract full text and parse raw items
    raw_text = _extract_text(pdf_path)
    items = _extract_raw_items(raw_text)

    # 2) Translate names to English in ONE batch using the separate module
    translator = MajorsTranslator()
    unique_names = [it["original_name"] for it in items]
    name_map = translator.translate_name_map(unique_names)

    # 3) Normalize payload to final schema
    payload: List[Dict[str, Any]] = []
    for it in items:
        on = it["original_name"].strip()
        en = str(name_map.get(on, on)).strip()
        # If English looks Hebrew (translation failed), fall back to original
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

    print(f"[majors_extractor] Wrote {len(payload)} majors -> {out_json}")
    return out_json
