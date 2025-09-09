# translator.py
"""
MajorsTranslator — translates original (Hebrew/native) major names to English.

Kept separate so you can change translation strategies (e.g., add a glossary or
post-process names) without reworking extraction or embeddings.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_TRANSLATE_TMPL = (
    "You are a precise academic translator. Translate each major name to a conventional English program name.\\n"
    "Return only a JSON object mapping original->english. No commentary.\\n\\n"
    "Examples:\\n"
    "- 'מדעי המחשב' -> 'Computer Science'\\n"
    "- 'הנדסת תעשייה וניהול' -> 'Industrial Engineering and Management'\\n\\n"
    "INPUT (JSON array of major names):\\n{names}\\n\\n"
    "OUTPUT (JSON object):"
)

class MajorsTranslator:
    """Translate a list of original names and return a mapping {original: english}."""

    def __init__(self, model_name: Optional[str] = None):
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        # Temperature 0 for stability/repeatability
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(_TRANSLATE_TMPL)
        self.parser = StrOutputParser()

    def translate_name_map(self, names: List[str]) -> Dict[str, str]:
        """Return a mapping {original_name: english_name}. Falls back to identity on errors."""
        names = [n for n in (names or []) if n]
        if not names:
            return {}
        payload = json.dumps(sorted(list(set(names))), ensure_ascii=False)
        chain = self.prompt | self.llm | self.parser
        out = chain.invoke({"names": payload})
        try:
            mapping = json.loads(out)
            if not isinstance(mapping, dict):
                raise ValueError("Expected a JSON object")
        except Exception:
            mapping = {n: n for n in names}  # identity fallback
        # Ensure coverage for all inputs
        for n in names:
            mapping.setdefault(n, n)
        return mapping
