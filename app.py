# app.py (updated to use majors_extractor + translator split)
import os, re
from dotenv import load_dotenv
from pdf_rag.ingest import PDFIngestor
from pdf_rag.query import HybridRAG
# UPDATED: use the new extractor (which internally calls translator)
from pdf_rag.majors_extractor import build_major_profiles

DATA_DIR = "data"
MAJORS_JSON = "extracted_majors.json"

def _find_majors_pdf() -> str:
    env_path = os.getenv("MAJORS_PDF")
    if env_path and os.path.exists(env_path):
        return env_path
    default_path = os.path.join(DATA_DIR, "majors.pdf")
    if os.path.exists(default_path):
        return default_path
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if not fname.lower().endswith(".pdf"):
                continue
            if re.search(r"(major|catalog|program|degree|תואר|חוג)", fname, flags=re.I):
                return os.path.join(DATA_DIR, fname)
    return ""

def _needs_build(src_pdf: str, out_json: str) -> bool:
    if not src_pdf or not os.path.exists(src_pdf):
        return False
    if not os.path.exists(out_json):
        return True
    try:
        return os.path.getmtime(out_json) < os.path.getmtime(src_pdf)
    except OSError:
        return True

def main():
    load_dotenv()
    ingestor = PDFIngestor(pdf_dir=DATA_DIR, meta_file="ingested_files.json")
    new_chunks = ingestor.ingest_new()
    rag = HybridRAG(
        persist_dir="chroma_db",
        collection_name="pdf_collection",
        embedding_model=os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        llm_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        k=3,
    )
    if new_chunks:
        print("[app] Upserting new chunks into Chroma...")
        rag.upsert_chunks(new_chunks)
        print("[app] Upsert done.")

    majors_pdf = _find_majors_pdf()
    if majors_pdf:
        if _needs_build(majors_pdf, MAJORS_JSON):
            print(f"[app] Building majors profiles from: {majors_pdf}")
            try:
                out = build_major_profiles(majors_pdf, out_json=MAJORS_JSON)
                print(f"[ok] Majors profiles ready -> {out}")
            except Exception as e:
                print(f"[warn] Failed to build majors profiles: {e}")
        else:
            print("[app] Majors profiles are up to date.")
        loaded = rag.load_majors_from_json(MAJORS_JSON)
        print(f"[app] Loaded {loaded} majors into memory.")
    else:
        print("[hint] No majors PDF found. Put one at data/majors.pdf or set MAJORS_PDF in .env")

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
            rag.run_interview()
            continue
        try:
            ans = rag.ask(q)
            print("\n--- Answer ---")
            print(ans)
        except Exception as e:
            print(f"[error] {e}")

if __name__ == "__main__":
    main()
