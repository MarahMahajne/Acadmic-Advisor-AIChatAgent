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
    Return the 50-criteria rubric as (key, description) pairs.
    The order here is the canonical order for embedding vectors.
    Scores are expected in range 0–5 (integers).
    """
    return [
        ("math_intensity", "Depth and frequency of mathematics."),
        ("statistics_prob", "Statistics/probability emphasis."),
        ("theoretical_focus", "Abstract/theoretical orientation."),
        ("hands_on_practice", "Applied/practical/experiential work."),
        ("programming_coding", "Amount of coding/software scripting (any language)."),
        ("software_engineering", "Software design, architecture, testing, DevOps."),
        ("data_science_ai", "Data science / ML / AI topics."),
        ("algorithms_systems", "Algorithms, operating systems, distributed systems."),
        ("networks_security", "Networking, cybersecurity, protocols."),
        ("hardware_electronics", "Circuits, embedded, digital/analog hardware."),
        ("physics_foundations", "Physics, mechanics, electromagnetism."),
        ("chemistry_emphasis", "General/organic/inorganic/physical chemistry."),
        ("biology_life_sciences", "Biology, genetics, physiology, bio-related areas."),
        ("earth_env_science", "Earth science, climate, ecology, environment."),
        ("sustainability", "Sustainable design/analysis, ESG, lifecycle thinking."),
        ("healthcare_clinical", "Clinical or healthcare systems exposure (non-patient)."),
        ("patient_interaction", "Direct patient/clinical interaction requirements."),
        ("lab_intensity", "Wet/physical lab work frequency and importance."),
        ("fieldwork_outdoor", "Fieldwork/outdoor data collection or site visits."),
        ("design_creativity", "Creative design, ideation, aesthetics."),
        ("studio_practice", "Studio-based courses (e.g., architecture/design/art)."),
        ("materials_manufacturing", "Materials science, manufacturing processes."),
        ("robotics_mechatronics", "Robotics, control, mechatronics."),
        ("ux_hci", "Human–computer interaction, UX research/design."),
        ("visual_arts", "Drawing, painting, photography, visual arts."),
        ("music_performance", "Music theory, performance, composition."),
        ("theater_performance", "Theatre, acting, stage production."),
        ("architecture_urban", "Architecture, urban design/planning."),
        ("civil_infrastructure", "Structures, transportation, water, geotech."),
        ("business_management", "Management, operations, organizational behavior."),
        ("finance_economics", "Finance, micro/macro economics, econometrics."),
        ("entrepreneurship", "Venture creation, innovation, product-market fit."),
        ("policy_government", "Public policy, governance, regulation."),
        ("law_ethics", "Law, ethics, compliance, standards."),
        ("psychology_behavior", "Psychology, behavioral science, cognition."),
        ("sociology_anthro", "Sociology, anthropology, social systems."),
        ("education_pedagogy", "Education theory, pedagogy, teaching practice."),
        ("communication_writing", "Academic/professional writing, documentation."),
        ("public_speaking", "Presentation and oral communication."),
        ("foreign_languages", "Foreign language requirement or emphasis."),
        ("history_heritage", "History, heritage, archival, historiography."),
        ("literature_philosophy", "Literature, philosophy, critical theory."),
        ("global_international", "International/global issues or perspectives."),
        ("quantitative_rigour", "Overall quantitative rigor across courses."),
        ("qualitative_methods", "Qualitative research methods and analysis."),
        ("research_methods", "Research design, methodology, inference."),
        ("teamwork_projects", "Team projects, collaboration, cross-functional work."),
        ("industry_internships", "Internships, co-ops, industry partnerships."),
        ("licensure_path", "Clear path to a professional license (e.g., PE, RN)."),
        ("workload_intensity", "Overall difficulty/time commitment."),
        ("employability_breadth", "Breadth of roles graduates can pursue."),
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
    with integer scores 0–5 for each criterion key, no commentary.
    """
    criteria = get_criteria()
    keys = [k for k, _ in criteria]
    rubric_lines = "\n".join([f"- {k}: {_desc}" for k, _desc in criteria])
    tmpl = (
        "You are scoring an academic MAJOR on a rubric of 50 criteria.\n"
        "Return ONLY a JSON object mapping each criterion key to an INTEGER 0–5.\n"
        "Rules:\n"
        "- 0 = not present at all; 5 = very strong emphasis; use integers only.\n"
        "- No extra text, no prose, no code fences.\n\n"
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

def build_major_embeddings(in_json: str = "extracted_majors.json",
                           out_json: str = "majors_embeddings.json") -> str:
    """
    Build embeddings for all majors by scoring each on the 50-criteria rubric.
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
    if not os.path.exists(in_json):
        raise FileNotFoundError(f"Input JSON not found: {in_json}")

    with open(in_json, "r", encoding="utf-8") as f:
        majors = json.load(f)
    if not isinstance(majors, list):
        raise ValueError("Input JSON must be a list of majors.")

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

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    print(f"[embed] Wrote {len(out_items)} majors -> {out_json}")
    return out_json
