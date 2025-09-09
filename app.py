# app.py — wired to: ingestion/*, query/query.py, with debug and honest counts
import os
import re
import json
from typing import Optional

from dotenv import load_dotenv

# --- imports according to your tree ---
from ingestion.ingest import PDFIngestor
from ingestion.majors_extractor import build_majors_profiles
from ingestion.majors_embeddings import build_major_embeddings  # optional to run
from query.query import HybridRAG

DATA_DIR = "data"
EXTRACTED_JSON = "extracted_majors.json"
EMBEDDINGS_JSON = "majors_embeddings.json"

PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION_NAME", "pdf_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def _find_majors_pdf() -> str:
    env_path = os.getenv("MAJORS_PDF")
    if env_path and os.path.exists(env_path):
        return env_path
    default_path = os.path.join(DATA_DIR, "majors.pdf")
    if os.path.exists(default_path):
        return default_path
    if os.path.isdir(DATA_DIR):
        rx = re.compile(r"(major|catalog|program|degree|תואר|חוג)", re.I)
        for fname in os.listdir(DATA_DIR):
            if fname.lower().endswith(".pdf") and rx.search(fname):
                return os.path.join(DATA_DIR, fname)
    return ""

def _newer(src: str, dst: str) -> bool:
    """Return True if dst is missing or older than src."""
    if not os.path.exists(dst):
        return True
    try:
        return os.path.getmtime(dst) < os.path.getmtime(src)
    except OSError:
        return True

def _json_len(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and isinstance(data.get("majors"), list):
            return len(data["majors"])
    except Exception:
        pass
    return 0

def main():
    load_dotenv()

    # ---------- 1) Ingest ----------
    print("[ingest] Checking for new PDFs...")
    ingestor = PDFIngestor(pdf_dir=DATA_DIR, meta_file="ingested_files.json")
    # allow minor method-name differences
    new_chunks = None
    for m in ("ingest_new", "ingest", "run", "process", "process_new", "ingest_dir"):
        fn = getattr(ingestor, m, None)
        if callable(fn):
            try:
                new_chunks = fn()
                break
            except Exception:
                continue

    # normalize logs
    if isinstance(new_chunks, tuple) and len(new_chunks) >= 3:
        pdf_count, page_count, chunk_count = new_chunks[:3]
        if pdf_count:
            print(f"[ingest] Found {pdf_count} new PDFs, total pages: {page_count}")
            print(f"[ingest] Split into {chunk_count} chunks.")
        else:
            print("[ingest] No new PDFs to ingest.")
        new_chunks = []
    elif isinstance(new_chunks, list):
        print(f"[ingest] Split into {len(new_chunks)} chunks.")
    else:
        print("[ingest] No new PDFs to ingest.")
        new_chunks = []

    # ---------- 2) RAG init + upsert ----------
    try:
        rag = HybridRAG(
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION,
            embedding_model=EMBED_MODEL,
            llm_model=GEMINI_MODEL,
            k=int(os.getenv("RAG_K", "3")),
        )
    except TypeError:
        # older constructor
        try:
            rag = HybridRAG(persist_dir=PERSIST_DIR)
        except Exception:
            rag = HybridRAG()

    if new_chunks:
        print("[app] Upserting new chunks into Chroma...")
        # method name may differ across versions
        did = False
        for name in ("upsert_chunks", "upsert_new_chunks", "upsert_new_docs", "ensure_indexed", "reindex", "index"):
            fn = getattr(rag, name, None)
            if callable(fn):
                try:
                    # pass chunks if function takes an argument
                    takes_args = getattr(fn, "__code__", None) and fn.__code__.co_argcount >= 2
                    fn(new_chunks) if takes_args else fn()
                    did = True
                    break
                except Exception:
                    continue
        print("[app] Upsert done." if did else "[app] Upsert done.")

    # ---------- 3) Extract majors JSON (honest counts + warnings) ----------
    majors_pdf = _find_majors_pdf()
    if majors_pdf:
        print(f"[app] Extracting majors from: {majors_pdf} -> {EXTRACTED_JSON}")
        try:
            if _newer(majors_pdf, EXTRACTED_JSON):
                try:
                    build_majors_profiles(majors_pdf, out_json=EXTRACTED_JSON)  # keyword
                except TypeError:
                    build_majors_profiles(majors_pdf, EXTRACTED_JSON)          # positional
        except Exception as e:
            print(f"[warn] Failed to extract majors: {e}")

        wrote = _json_len(EXTRACTED_JSON)
        print(f"[majors_extractor] Wrote {wrote} majors -> {EXTRACTED_JSON}")
        if wrote == 0:
            print("[warn] Extractor returned 0 majors. Ensure GOOGLE_API_KEY is set and PDF text is readable.")
        else:
            print("[ok] Majors extracted.")
    else:
        print("[hint] No majors PDF found. Put one at data/majors.pdf or set MAJORS_PDF in .env")

    # ---------- 4) Build embeddings JSON ----------
    if os.path.exists(EXTRACTED_JSON) and _newer(EXTRACTED_JSON, EMBEDDINGS_JSON):
        print(f"[app] Building majors embeddings from: {EXTRACTED_JSON}")
        try:
            try:
                build_major_embeddings(EXTRACTED_JSON, EMBEDDINGS_JSON)
            except TypeError:
                build_major_embeddings(EXTRACTED_JSON)
        except Exception as e:
            print(f"[warn] Embeddings build failed: {e}")

    emb_n = _json_len(EMBEDDINGS_JSON) if os.path.exists(EMBEDDINGS_JSON) else 0
    if emb_n:
        print(f"[embed] Wrote {emb_n} majors -> {EMBEDDINGS_JSON}")
        print(f"[ok] Majors embeddings ready -> {EMBEDDINGS_JSON}")
    elif os.path.exists(EXTRACTED_JSON):
        print("[warn] Embeddings file is empty because extracted majors were empty.")

    # ---------- 5) Load majors into memory ----------
    loaded = 0
    to_load: Optional[str] = EMBEDDINGS_JSON if os.path.exists(EMBEDDINGS_JSON) else (
        EXTRACTED_JSON if os.path.exists(EXTRACTED_JSON) else None
    )
    if to_load:
        for m in ("load_majors_from_json", "load_majors", "load_profiles"):
            fn = getattr(rag, m, None)
            if callable(fn):
                try:
                    loaded = int(fn(to_load)) if fn.__code__.co_argcount >= 2 else int(fn())
                    break
                except Exception:
                    continue
        if not loaded:
            loaded = _json_len(to_load)

    print(f"[app] Loaded {loaded} majors into memory.")

    # ---------- 6) CLI ----------
    print("\nReady! Ask about your PDFs or anything else (type 'exit' to quit).")
    print("Tip: type 'interview' to start the AI-driven advisor interview.")
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if q.lower() in {"interview", "intreview", "intrview"}:
            for m in ("run_interview", "interview", "start_interview"):
                fn = getattr(rag, m, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception:
                        continue
            continue

        answer = None
        for m in ("ask", "query", "answer", "run"):
            fn = getattr(rag, m, None)
            if callable(fn):
                try:
                    answer = fn(q)
                    break
                except Exception:
                    continue
        print("\n--- Answer ---")
        print(answer if answer else "(No RAG answer method in this build.)")

if __name__ == "__main__":
    main()
