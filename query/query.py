# query/query.py
from __future__ import annotations

import os
import json
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# ---- Embeddings & Vector store ----
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback


# ---- LLM (Gemini) ----
from langchain_google_genai import ChatGoogleGenerativeAI

# ---- Prompts ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ========================== Data types ==========================

@dataclass
class MajorProfile:
    original_name: str          # exactly as in the PDF
    english_name: str           # translated to English for embedding
    keywords: List[str]
    sample_courses: List[str]
    source: str = "majors.pdf"


# ========================== RAG Engine ==========================

class HybridRAG:
    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "pdf_collection",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gemini-2.0-flash",
        k: int = 3,
    ):
        self.k = k
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # --- Embeddings ---
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except TypeError:
            # older langchain versions
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
            )

        # --- Vector store ---
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )

        # --- Retriever (uses k) ---
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )

        # --- LLM (Gemini) ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Put it in your environment or .env file.")
        # Keep defaults to avoid deprecation warnings; system prompts are added via ChatPromptTemplate
        self.llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.2)
        self.output_parser = StrOutputParser()

        # Majors memory
        self.majors: List[MajorProfile] = []
        self._major_vectors: List[Tuple[str, List[float]]] = []  # (english_name, vector)
        self._name_map: Dict[str, str] = {}  # english_name -> original_name

    # ===================== Ingestion / Upsert =====================

    def upsert_chunks(self, chunks: List) -> None:
        """Insert split LangChain Documents into Chroma."""
        if not chunks:
            return
        self.vectorstore.add_documents(chunks)

    # ========================= Grounded Q&A ========================

    def retrieve(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search(query, k=k)

    def ask(self, question: str, k: int = 5) -> str:
        
        # 1. REMOVE THE RETRIEVAL STEP
        # The following line is now commented out or removed.
        # docs = self.retrieve(question, k=k)
        # context = "\n\n".join(
        #     f"[{d.metadata.get('source','?')} p{d.metadata.get('page', '?')}] {d.page_content[:1200]}"
        #     for d in docs
        # )
        
        # 2. SET THE CONTEXT TO BE EMPTY
        # We set the context variable to an empty string to ensure it's not included in the prompt.
        context = ""

        system = (
            "You are an academic advisor assistant. Your primary goal is to answer the user's question, "
            "relying on your general knowledge. You will not be provided with any external context.\n\n"
            "LANGUAGE POLICY:\n"
            "• CRITICAL: Respond ONLY in the language of the user's question. If the user asks in English, respond in English. If they ask in Hebrew, respond in Hebrew. If they ask in Arabic, respond in Arabic.\n"
            "• Do NOT use any other language in your response.\n"
            "• Do NOT echo or quote the question or add any markers/tags. Output only the answer text."
        )

        tmpl = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human",
             "User question:\n{q}\n\n"
             "Answer now, strictly following the LANGUAGE POLICY.")
        ])
        chain = tmpl | self.llm | self.output_parser
        
        # 3. INVOKE THE CHAIN WITHOUT CONTEXT
        # The 'c' parameter is no longer needed since context is an empty string.
        return chain.invoke({"q": question}).strip()

    

    # ---------- Load majors JSON built earlier into memory ----------

    def load_majors_from_json(self, path: str = "extracted_majors.json") -> int:
        """
        Load majors from a JSON file written by the builder.
        Ensures each major has an English name, pre-embeds them, and builds a name map.
        Returns the number of majors loaded.
        """
        if not os.path.exists(path):
            self.majors, self._major_vectors, self._name_map = [], [], {}
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f) or []
        except Exception as e:
            print(f"[majors] Failed to read {path}: {e}")
            self.majors, self._major_vectors, self._name_map = [], [], {}
            return 0

        majors: List[MajorProfile] = []
        for item in raw:
            majors.append(
                MajorProfile(
                    original_name=str(item.get("original_name", "")).strip(),
                    english_name=str(item.get("english_name", "")).strip(),
                    keywords=[str(x).strip() for x in item.get("keywords", []) if str(x).strip()],
                    sample_courses=[str(x).strip() for x in item.get("sample_courses", []) if str(x).strip()],
                    source=str(item.get("source", "majors.pdf")),
                )
            )

        # Keep names as in JSON (already English). If needed you can call _ensure_english_names here.
        self.majors = majors

        # Pre-embed for later matching (uses english_name + keywords + courses)
        texts, names = [], []
        for m in self.majors:
            profile_text = " | ".join([
                m.english_name,
                " ".join(m.keywords[:10]),
                " ".join(m.sample_courses[:30]),
            ])
            texts.append(profile_text)
            names.append(m.english_name)
        self._major_vectors = self.embeddings.embed_documents(texts) if texts else []
        self._major_vectors = list(zip(names, self._major_vectors)) if texts else []
        self._name_map = {m.english_name: m.original_name for m in self.majors}

        return len(self.majors)

   
    # ==================== Interview & Reco engine ====================

    def run_interview(self, max_turns: int = 10):
        """
        Dynamic interview (no hardcoded bank). The model asks ONE question per turn.
        It is instructed to speak in the same language as the student's latest message,
        defaulting to English if unclear. Repeat-guard prevents stuck loops.
        """
        print("\n--- Academic Advisor Interview (dynamic) ---")
        history: List[Tuple[str, str]] = []  # ("ai" or "user", text)
        asked_questions: List[str] = []
        turns = 0

        while True:
            majors_hint = ", ".join(m.english_name for m in self.majors[:20]) if self.majors else ""
            topics_covered = self._extract_topics_covered(history)

            nxt = self._interview_turn(
                history,
                majors_hint=majors_hint,
                asked_questions=asked_questions,
                topics_covered=topics_covered
            )

            if nxt.upper().startswith("RECOMMEND:"):
                answers = self._normalize_prefs_from_history(history)
                block = self._final_recommendations_block_from_answers(answers)
                print(block)
                return

            question = nxt
            if not question.upper().startswith("QUESTION:"):
                nxt2 = self._interview_turn(
                    history,
                    majors_hint=majors_hint,
                    asked_questions=asked_questions,
                    topics_covered=topics_covered,
                    force_question=True
                )
                question = nxt2 if nxt2.upper().startswith("QUESTION:") else "QUESTION: On a scale of 0–5, how much do you enjoy programming/software development?"

            q_text = question.split(":", 1)[1].strip() if ":" in question else question.strip()

            # Repeat guard
            norm_q = re.sub(r"\s+", " ", q_text.lower()).strip()
            if asked_questions and norm_q == re.sub(r"\s+", " ", asked_questions[-1].lower()).strip():
                nxt3 = self._interview_turn(
                    history,
                    majors_hint=majors_hint,
                    asked_questions=asked_questions,
                    topics_covered=topics_covered,
                    avoid_repeat_of=q_text
                )
                if nxt3.upper().startswith("QUESTION:"):
                    q_text = nxt3.split(":", 1)[1].strip()
                else:
                    q_text = self._fallback_question(topics_covered)

            print("\nAI:", q_text)
            user_ans = input("You: ").strip()
            if user_ans.lower() in ("stop", "exit", "quit"):
                print("Interview stopped.")
                return

            history.append(("ai", q_text))
            history.append(("user", user_ans))
            asked_questions.append(q_text)
            turns += 1

            if turns >= max_turns:
                answers = self._normalize_prefs_from_history(history)
                block = self._final_recommendations_block_from_answers(answers)
                print(block)
                return

    def _interview_turn(
        self,
        history: List[Tuple[str, str]],
        majors_hint: str = "",
        asked_questions: Optional[List[str]] = None,
        topics_covered: Optional[set] = None,
        force_question: bool = False,
        avoid_repeat_of: Optional[str] = None,
    ) -> str:
        """One-turn driver. Model must speak in the student's language; if unclear, default to English."""
        asked_questions = asked_questions or []
        topics_covered = topics_covered or set()

        # Last user utterance (if any)
        last_user = ""
        for role, text in reversed(history):
            if role == "user":
                last_user = text.strip()
                break

        system = (
            "You are an academic advisor interviewing a student to recommend a major. "
            "Ask brief, clear questions ONE at a time.\n\n"
            "LANGUAGE POLICY:\n"
            "- Detect the student's language ONLY from the student's latest message shown below (plain text).\n"
            "- If that latest message is empty or unclear, DEFAULT TO ENGLISH.\n"
            "- Reply ONLY in that detected language (or English if unclear). Do NOT use any other language.\n"
            "- Do not repeat questions you've already asked.\n"
            "When enough info is gathered, output a single line beginning with 'RECOMMEND:'\n"
        )
        if majors_hint:
            system += f"Preferred majors set: {majors_hint}. "
        if force_question:
            system += "IMPORTANT: Output a single line that begins with 'QUESTION:' only. "
        if avoid_repeat_of:
            system += f"Do NOT ask anything similar to this: '{avoid_repeat_of}'. "

        asked_blob = "; ".join(asked_questions[-8:]) if asked_questions else "none"
        covered_blob = ", ".join(sorted(topics_covered)) if topics_covered else "none"

        # Few-shot anchors to prevent drifting into other languages
        examples = [
            ("human", "Student (latest): hi"),
            ("ai", "QUESTION: What subjects do you enjoy studying the most?"),
            ("human", "Student (latest): שלום"),
            ("ai", "QUESTION: אילו מקצועות את/ה הכי אוהב/ת ללמוד?"),
            ("human", "Student (latest): مرحبا"),
            ("ai", "QUESTION: ما المواد التي تستمتع بدراستها أكثر؟"),
        ]

        chat_msgs = [("system", system)]
        if last_user:
            chat_msgs.append(("system", f"Student (latest, for language detection): {last_user}"))
        for role, text in history:
            chat_msgs.append(("ai" if role == "ai" else "human", text))
        chat_msgs.append(("human",
            "Do not repeat yourself. Recently asked: {asked}.\n"
            "Topics already covered: {covered}.\n"
            "Your next single line (either 'QUESTION: ...' or 'RECOMMEND: ...'):")
        )

        # Insert the anchors just before the final instruction
        chat_msgs = chat_msgs[:1] + examples + chat_msgs[1:]

        chat = ChatPromptTemplate.from_messages(chat_msgs)
        msg = chat.format(asked=asked_blob, covered=covered_blob)
        out = self.llm.invoke(msg).content.strip()
        return out

    def _fallback_question(self, topics_covered: set) -> str:
        options = [
            ("math", "On a scale of 0–5, how much math intensity are you comfortable with?"),
            ("coding", "On a scale of 0–5, how much do you enjoy programming/software development?"),
            ("applied", "On a scale of 0–5, do you prefer applied learning (5) or theory (0)?"),
            ("labs", "How important are lab/practical sessions for you? (low/medium/high)"),
            ("ai_ml", "Are you interested in AI/ML and data science? (yes/no/maybe)"),
            ("software", "Are you interested in software/web/app development? (yes/no/maybe)"),
            ("hardware", "Are you interested in hardware/electronics/embedded systems? (yes/no/maybe)"),
            ("bio", "Do you like biology-related topics? (like/dislike/neutral)"),
            ("chem", "Do you like chemistry-related topics? (like/dislike/neutral)"),
            ("phys", "On a scale of 0–5, how much do you like physics?"),
        ]
        for key, q in options:
            if key not in topics_covered:
                return q
        return "Based on what we discussed, please summarize your preferences so I can recommend majors."

    def _extract_topics_covered(self, history: List[Tuple[str, str]]) -> set:
        text = " ".join(t.lower() for _, t in history)
        covered = set()
        if re.search(r"\b(math|calculus|algebra)\b", text) or re.search(r"\b[0-5]\s*/\s*5\b", text):
            covered.add("math")
        if re.search(r"\b(coding|programming|software|developer|dev)\b", text):
            covered.add("coding")
        if re.search(r"\b(applied|theory|theoretical|hands[- ]?on)\b", text):
            covered.add("applied")
        if re.search(r"\b(lab|labs|laboratory|practical)\b", text):
            covered.add("labs")
        if re.search(r"\b(ai|machine learning|ml|data science)\b", text):
            covered.add("ai_ml")
        if re.search(r"\b(web|app|frontend|backend|software)\b", text):
            covered.add("software")
        if re.search(r"\b(hardware|electronics|embedded)\b", text):
            covered.add("hardware")
        if re.search(r"\b(bio|biology)\b", text):
            covered.add("bio")
        if re.search(r"\b(chem|chemistry)\b", text):
            covered.add("chem")
        if re.search(r"\b(physics|physic)\b", text):
            covered.add("phys")
        return covered

    def _normalize_prefs_from_history(self, history: List[Tuple[str, str]]) -> Dict[str, str]:
        transcript = "\n".join([f"{'AI' if r=='ai' else 'Student'}: {t}" for r, t in history])
        schema = {
            "math_level": "0-5 int",
            "coding_interest": "0-5 int",
            "applied_theory": "0-5 int (0=theory, 5=applied)",
            "labs": "string (low/medium/high)",
            "ai_ml": "string (yes/no/maybe)",
            "software": "string (yes/no/maybe)",
            "hardware": "string (yes/no/maybe)",
            "bio": "string (like/dislike/neutral)",
            "chem": "string (like/dislike/neutral)",
            "phys": "0-5 int"
        }
        tmpl = ChatPromptTemplate.from_messages([
            ("system", "Extract a compact preference profile from the conversation. Return JSON ONLY, no commentary."),
            ("human",
             "Conversation transcript:\n{tx}\n\n"
             "Return valid JSON with these keys (fill best guess if unspecified):\n{schema}\n\nJSON only:"
            )
        ])
        prompt = tmpl.format(tx=transcript, schema=json.dumps(schema, ensure_ascii=False))
        out = self.llm.invoke(prompt).content.strip()
        out = re.sub(r"^```(?:json)?", "", out, flags=re.I).strip()
        out = re.sub(r"```$", "", out).strip()

        try:
            data = json.loads(out)
        except Exception:
            data = {}

        def _ival(val, default=3):
            try:
                return max(0, min(5, int(round(float(val)))))
            except Exception:
                return default

        return {
            "math_level": _ival(data.get("math_level", 3)),
            "coding_interest": _ival(data.get("coding_interest", 3)),
            "applied_theory": _ival(data.get("applied_theory", 3)),
            "labs": str(data.get("labs", "medium")),
            "ai_ml": str(data.get("ai_ml", "maybe")),
            "software": str(data.get("software", "maybe")),
            "hardware": str(data.get("hardware", "maybe")),
            "bio": str(data.get("bio", "neutral")),
            "chem": str(data.get("chem", "neutral")),
            "phys": _ival(data.get("phys", 3)),
        }

    # ===================== Recommendation core =====================

    def recommend_from_interview(self, answers: Dict[str, str], top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Map normalized answers to an English profile text, embed it,
        and return top_k: (english_major_name, score, rationale in English).
        """
        if not self._major_vectors:
            return []

        prefs = [
            f"Math level: {answers.get('math_level','unspecified')}/5",
            f"Coding interest: {answers.get('coding_interest','unspecified')}/5",
            f"Applied vs theory: {answers.get('applied_theory','unspecified')}/5",
            f"Labs importance: {answers.get('labs','unspecified')}",
            f"AI/ML: {answers.get('ai_ml','unspecified')}",
            f"Software/web: {answers.get('software','unspecified')}",
            f"Hardware/embedded: {answers.get('hardware','unspecified')}",
            f"Biology emphasis: {answers.get('bio','unspecified')}",
            f"Chemistry emphasis: {answers.get('chem','unspecified')}",
            f"Physics: {answers.get('phys','unspecified')}/5",
        ]
        profile_text = " | ".join(prefs)

        v = self.embeddings.embed_query(profile_text)

        def cos(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            return dot / (na*nb + 1e-12)

        sims = [(name, cos(v, vec)) for name, vec in self._major_vectors]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_k]

        results = []
        for name, sc in top:
            original = name
            rationale = self._explain_major_choice(name, original, profile_text)
            results.append((name, float(sc), rationale))
        return results

    def _final_recommendations_block_from_answers(self, answers: Dict[str, str]) -> str:
        recs = self.recommend_from_interview(answers, top_k=3)
        if not recs:
            return "[warn] Could not compute recommendations. Make sure extracted_majors.json was built and loaded."

        lines = ["\n--- Final Recommendations (college majors only) ---"]
        for i, (name_en, score, why) in enumerate(recs, start=1):
            sim_pct = f"{score*100:.1f}%"
            lines.append(f"{i}. {name_en}  (match {sim_pct})\n   Why: {why}")
        return "\n".join(lines)

    # =================== Major context & explanation ===================

    def _retrieve_major_context(self, english_name: str, original_name: Optional[str], extra_hint: str = "") -> List[Any]:
        q = " ".join(filter(None, [english_name, original_name, extra_hint]))
        try:
            docs = [d for d in self.vectorstore.similarity_search(q, k=4)]
        except Exception:
            docs = []
        return docs

    def _explain_major_choice(self, english_name: str, original_name: str, student_profile_text: str) -> str:
        hint = ""
        for m in self.majors:
            if m.english_name == english_name:
                hint = ", ".join((m.keywords or m.sample_courses)[:6])
                break

        docs = self._retrieve_major_context(english_name, original_name, hint)
        context = "\n\n".join(d.page_content[:1000] for d in docs) if docs else ""

        prompt = ChatPromptTemplate.from_template(
            "You are an Academic Advisor. Use the PDF context if relevant; otherwise, provide a general rationale. "
            "Write 2–3 concise sentences explaining why the following major fits the student.\n\n"
            "LANGUAGE: Respond ONLY in English. Keep the major name in English.\n\n"
            "Major (English name): {major_en}\n"
            "Student profile: {profile}\n"
            "PDF context (may be empty):\n{ctx}\n\n"
            "Return only the explanation in English."
        )
        chain = prompt | self.llm | self.output_parser
        why = chain.invoke({"major_en": english_name, "profile": student_profile_text, "ctx": context}).strip()

        if docs:
            srcs = sorted({d.metadata.get("source") for d in docs if d.metadata.get("source")})
            if srcs:
                why += "\nSources: " + ", ".join(srcs)
        return why
