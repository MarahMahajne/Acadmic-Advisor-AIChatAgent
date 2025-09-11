# ingestor/majors_embeddings.py
"""
Majors embeddings builder (diverse 50-criteria rubric).

What this module does
---------------------
• Loads majors extracted earlier (from `extracted_majors.json`), which should have:
    [
      {
        "original_name": str,
        "english_name": str,
        "keywords": [str],
        "sample_courses": [str],
        "source": str
      },
      ...
    ]
• Uses Gemini to score each major on a broad set of **50 cross-disciplinary criteria**
  (STEM, humanities, arts, health, environment, business, policy, design, etc.).
• Saves a clean embeddings file `majors_embeddings.json` that your recommendation
  code can use for similarity against a student’s preference vector.

Environment variables
---------------------
- GOOGLE_API_KEY  (required)
- GEMINI_MODEL    (default: "gemini-2.0-flash")

Public entrypoint
-----------------
- build_major_embeddings(in_json="extracted_majors.json",
                         out_json="majors_embeddings.json")
"""

from __future__ import annotations

import os, json, re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# --------------------------- Criteria ---------------------------

def get_criteria() -> List[Tuple[str, str]]:
    """
    Return the rubric for cognitive + language style.
    These are the 24 criteria we defined for mapping majors.
    Scores are expected in range 0–100.
    """
    return [
        ("data_analysis", "יכולת ניתוח נתונים והסקת מסקנות."),
        ("quantitative_reasoning", "חשיבה כמותית ושימוש במודלים מתמטיים."),
        ("logical_problem_solving", "פתרון בעיות לוגיות מורכבות."),
        ("pattern_recognition", "זיהוי דפוסים ומגמות במידע."),
        ("systems_thinking", "יכולת ראיית תמונה כוללת והבנת מערכות."),
        ("risk_decision", "שקלול סיכונים וקבלת החלטות בתנאי אי־ודאות."),
        ("strategic_thinking", "חשיבה אסטרטגית ארוכת טווח."),
        ("cognitive_flexibility", "גמישות מחשבתית והתמודדות עם שינויים."),
        ("curiosity_interdisciplinary", "סקרנות ויכולת לחבר בין תחומים שונים."),
        ("creative_solutions", "יכולת יצירת פתרונות מקוריים."),
        ("innovation_openness", "נכונות לניסוי וטעייה, פתיחות לחדשנות."),
        ("opportunity_recognition", "ראיית הזדמנויות והפיכתן לפעולה."),
        ("social_sensitivity", "רגישות למצבים חברתיים והבנת נקודות מבט שונות."),
        ("teamwork", "עבודה בצוות, שיתוף פעולה ויכולת למידה משותפת."),
        ("communication_mediation", "יכולת תקשורת בין־אישית וגישור."),
        ("ethical_awareness", "מודעות לאתיקה ודילמות ערכיות."),
        ("psychological_interest", "עניין במניעים ותהליכים פסיכולוגיים."),
        ("theoretical_patience", "סבלנות ללמידה תאורטית לצד יישום."),
        ("attention_to_detail", "קשב לדקויות בתקשורת ובמידע."),
        ("leadership", "לקיחת אחריות, הובלה וניהול יוזמות."),
        # --- הוספות שפה ---
        ("hebrew_proficiency", "ידע והבנה בעברית אקדמית/מקצועית."),
        ("english_proficiency", "ידע והבנה באנגלית אקדמית/מקצועית."),
        ("hebrew_expression", "יכולת התבטאות בעברית בכתב ובעל פה."),
        ("english_expression", "יכולת התבטאות באנגלית בכתב ובעל פה."),
    ]


# --------------------------- LLM Utils ---------------------------

def _make_llm(model_env: str = "GEMINI_MODEL",
              default_model: str = "gemini-2.0-flash",
              temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Build a Gemini chat model instance; temperature=0 for deterministic rubric scoring.
    Requires GOOGLE_API_KEY. Model name may be overridden via GEMINI_MODEL.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    model_name = os.getenv(model_env, default_model)
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _safe_json_loads(s: str, default: Any = None) -> Any:
    """
    Robust JSON parse with simple repairs (strip code fences, trailing commas, single quotes).
    Returns `default` on failure.
    """
    if not s:
        return default
    s = re.sub(r"^```(?:json)?", "", s.strip())
    s = re.sub(r"```$", "", s.strip())
    try:
        return json.loads(s)
    except Exception:
        # salvage inner JSON
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        s2 = s2.replace("'", '"')
        try:
            return json.loads(s2)
        except Exception:
            return default


# --------------------------- Scoring Prompt ---------------------------

def _build_scoring_prompt() -> ChatPromptTemplate:
    """
    Create the scoring prompt that asks Gemini to output a JSON object
    with integer scores 0–100 for each criterion key, no commentary.
    This is used to build embedding vectors for majors and persons,
    so consistency across all criteria is critical.
    """
    criteria = get_criteria()  # 24 or 50 criteria, depending on what you want
    keys = [k for k, _ in criteria]
    rubric_lines = "\n".join([f"- {k}: {_desc}" for k, _desc in criteria])
    tmpl = (
        "You are evaluating an academic MAJOR (or a PERSON profile) "
        "on a rubric of cognitive, behavioral, and language criteria.\n"
        "Return ONLY a JSON object mapping each criterion key to an INTEGER 0–100.\n"
        "\n"
        "Rules:\n"
        "- 0 = not present at all.\n"
        "- 100 = extremely strong emphasis.\n"
        "- Use only integers between 0 and 100.\n"
        "- This JSON will be converted into an embedding vector, so all keys must be present.\n"
        "- No commentary, no prose, no code fences — JSON only.\n\n"
        f"Criteria and meanings:\n{rubric_lines}\n\n"
        "MAJOR NAME (English): {name}\n"
        "SAMPLE COURSES (as-is, may contain Hebrew): {courses}\n"
        "KEYWORDS: {keywords}\n\n"
        "JSON object with these keys only:\n"
        f"{keys}\n"
    )
    return ChatPromptTemplate.from_template(tmpl)


# --------------------------- Core Scoring ---------------------------

def _score_one_major(item: Dict[str, Any],
                     llm: ChatGoogleGenerativeAI,
                     prompt: ChatPromptTemplate) -> Dict[str, int]:
    """
    Score a single major dict (expects fields: english_name, sample_courses, keywords).
    Returns a dict {criterion_key: int_score_0_to_5}.
    """
    name = item.get("english_name") or item.get("original_name") or ""
    courses = item.get("sample_courses", [])
    keywords = item.get("keywords", [])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "name": name,
        "courses": json.dumps(courses, ensure_ascii=False),
        "keywords": json.dumps(keywords, ensure_ascii=False),
    })
    scores = _safe_json_loads(raw, default={})
    # Coerce to 0-5 ints and ensure all keys exist
    result: Dict[str, int] = {}
    for k, _ in get_criteria():
        v = scores.get(k, 0)
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        iv = max(0, min(5, iv))
        result[k] = iv
    return result


# --------------------------- Public API ---------------------------

def build_majors_embeddings(in_json: str = "extracted_majors.json",
                           out_json: str = "majors_embeddings.json") -> str:
    """
    Build embeddings for all majors by scoring each on the 50-criteria rubric.
    Skips rebuilding if `out_json` already exists and contains a non-empty list.

    Input:
        in_json  – path to the majors list produced by majors_extractor.py
    Output:
        out_json – path to JSON list with scores, one object per major:
            {
              "original_name": str,
              "english_name": str,
              "scores": {criterion_key: int(0..5)},
              "source": str
            }
    Returns the output path.
    """
    # --- Skip if embeddings already exist and look valid ---
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list) and len(existing) > 0:
                print(f"[skip] {out_json} already exists with {len(existing)} majors. Skipping embeddings build.")
                return out_json
            else:
                print(f"[warn] {out_json} exists but is empty/invalid. Rebuilding...")
        except Exception as e:
            print(f"[warn] Failed to read existing {out_json} ({e}). Rebuilding...")

    # --- Validate input majors JSON ---
    if not os.path.exists(in_json):
        raise FileNotFoundError(f"Input JSON not found: {in_json}")

    with open(in_json, "r", encoding="utf-8") as f:
        majors = json.load(f)
    if not isinstance(majors, list):
        raise ValueError("Input JSON must be a list of majors.")

    # --- Build embeddings/scores ---
    llm = _make_llm(temperature=0.0)
    prompt = _build_scoring_prompt()

    out_items: List[Dict[str, Any]] = []
    for i, item in enumerate(majors, 1):
        try:
            scores = _score_one_major(item, llm, prompt)
        except Exception as e:
            print(f"[warn] Scoring failed for '{item.get('english_name') or item.get('original_name')}' :: {e}")
            scores = {k: 0 for k, _ in get_criteria()}
        out_items.append({
            "original_name": item.get("original_name", ""),
            "english_name": item.get("english_name", ""),
            "scores": scores,
            "source": item.get("source", ""),
        })
        if i % 5 == 0:
            print(f"[embed] Scored {i}/{len(majors)} majors...")

    # --- Write output ---
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"[embed] Wrote {len(out_items)} majors -> {out_json}")
    return out_json
