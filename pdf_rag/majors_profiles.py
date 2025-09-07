# pdf_rag/majors_profiles.py
import os, json, math
import pdfplumber
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ---- Shared criteria rubric (0–5 each) ----
CRITERIA = [
    ("math_intensity", "How math-heavy the curriculum is."),
    ("programming_coding", "Amount of coding / software development."),
    ("theory_to_practice", "0 means mostly theory; 5 means strongly hands-on/applied."),
    ("lab_work", "Frequency/importance of labs or practical sessions."),
    ("data_ai_focus", "Focus on data science / AI / ML."),
    ("software_engineering_apps", "Focus on software/app/web development."),
    ("hardware_electronics", "Focus on hardware/electronics/embedded."),
    ("biology_emphasis", "Biology-related emphasis."),
    ("chemistry_emphasis", "Chemistry-related emphasis."),
    ("physics_emphasis", "Physics-related emphasis."),
    ("writing_communication", "Writing, presentations, communication load."),
    ("teamwork_collaboration", "Team projects, collaboration emphasis."),
    ("creativity_design", "Design/creative components (UX, product, design projects)."),
    ("business_component", "Business/entrepreneurship/management exposure."),
]

# Single source of truth for English-only system constraint
SYSTEM_EN = (
    "You are extracting/assessing academic catalog information. "
    "Your ENTIRE OUTPUT must be in ENGLISH ONLY. "
    "Return STRICT JSON ONLY with NO extra commentary, prose, prefaces, or code fences."
)

def _llm():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.2,
        convert_system_message_to_human=True
    ), StrOutputParser()

def _read_pdf_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)

def _extract_majors(raw_text: str) -> List[Dict[str, Any]]:
    """
    Ask Gemini to extract a structured list of majors with an optional course list.
    Expected output: JSON like:
    {"majors":[{"name":"Computer Science","courses":["...","..."], "excerpt":"..."}, ...]}
    (ENGLISH ONLY)
    """
    llm, parser = _llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_EN),
        ("human",
         "From the text below, identify all majors offered by THE COLLEGE, and for each, "
         "list 8–20 representative courses (if visible) and provide a short excerpt (2–5 lines) "
         "summarizing what the major covers.\n\n"
         "IMPORTANT:\n"
         "- Use ENGLISH ONLY (even if the source text is not fully English).\n"
         "- Return STRICT JSON ONLY with the exact shape:\n"
         '  {"majors":[{"name":"...","courses":["..."],"excerpt":"..."}]}\n'
         "- Do NOT include any non-English words.\n\n"
         "Text:\n{txt}")
    ])
    chain = prompt | llm | parser

    out = chain.invoke({"txt": raw_text})

    # Robust JSON load
    def _safe_load(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            # try to salvage a JSON object or array substring
            start_obj, start_arr = s.find("{"), s.find("[")
            starts = [i for i in [start_obj, start_arr] if i >= 0]
            if not starts:
                return {}
            start = min(starts)
            stack = []
            for i, ch in enumerate(s[start:], start=start):
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and ch == stack[-1]:
                        stack.pop()
                        if not stack:
                            try:
                                return json.loads(s[start:i+1])
                            except Exception:
                                break
            return {}

    data = _safe_load(out)
    majors = data.get("majors", []) if isinstance(data, dict) else []

    # basic cleanup
    cleaned = []
    seen = set()
    for m in majors:
        name = (m.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        courses = m.get("courses") or []
        if isinstance(courses, list):
            courses = [str(c).strip() for c in courses if str(c).strip()]
        else:
            # sometimes models return a comma-joined string; normalize
            courses = [c.strip() for c in str(courses).split(",") if c.strip()]
        excerpt = (m.get("excerpt") or "").strip()[:1200]
        cleaned.append({"name": name, "courses": courses, "excerpt": excerpt})

    # defensive fallback
    if not cleaned:
        return [{"name": "General Major", "courses": [], "excerpt": raw_text[:2000]}]

    return cleaned

def _score_major(major: Dict[str, Any]) -> Dict[str, float]:
    """
    Score a single major on the CRITERIA with Gemini. Output STRICT JSON {criterion: 0-5,...}.
    (ENGLISH ONLY)
    """
    llm, parser = _llm()
    rubric_lines = "\n".join([f'- "{cid}": {desc}' for cid, desc in CRITERIA])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_EN),
        ("human",
         "Rate the following academic major on multiple criteria from 0–5 "
         "(0 = none, 5 = very strong).\n\n"
         "Return STRICT JSON ONLY with exactly these keys (integers 0–5):\n"
         "{rubric}\n\n"
         "Major name: {name}\n"
         "Representative courses: {courses}\n"
         "Excerpt: {excerpt}\n"
         "JSON:")
    ])
    chain = prompt | llm | parser

    out = chain.invoke({
        "rubric": rubric_lines,
        "name": major["name"],
        "courses": ", ".join(major.get("courses", []))[:2000],
        "excerpt": major.get("excerpt", "")[:2000],
    })

    # Parse & clamp
    try:
        scores = json.loads(out)
    except Exception:
        scores = {cid: 0 for cid, _ in CRITERIA}

    vec: Dict[str, float] = {}
    for cid, _ in CRITERIA:
        v = scores.get(cid, 0)
        try:
            v = int(round(float(v)))
        except Exception:
            v = 0
        vec[cid] = max(0, min(5, v))
    return vec

def build_major_profiles(pdf_path: str, out_json: str = "majors_profiles.json") -> str:
    """
    Extract majors from the catalog PDF, score them, and save to JSON (ENGLISH ONLY).
    """
    raw = _read_pdf_text(pdf_path)
    majors = _extract_majors(raw)
    for m in majors:
        m["scores"] = _score_major(m)
    payload = {
        "criteria": [{"id": cid, "desc": desc} for cid, desc in CRITERIA],
        "majors": majors,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_json

# ---------- utilities for recommendation ----------
def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys: return 0.0
    num = sum(a[k]*b[k] for k in keys)
    da = math.sqrt(sum(a[k]*a[k] for k in keys))
    db = math.sqrt(sum(b[k]*b[k] for k in keys))
    return 0.0 if da == 0 or db == 0 else num/(da*db)

def top_k_majors(student_vec: Dict[str, float], majors_blob: Dict[str, Any], k: int = 3):
    rows = []
    for m in majors_blob.get("majors", []):
        sim = cosine(student_vec, m.get("scores", {}))
        rows.append((sim, m))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:k]
