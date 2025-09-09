# ingestor/ingest.py
import os
import json
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFIngestor:
    """
    Handles ingestion of PDF files into the system:
      - Detects new PDF files in a directory.
      - Loads their content (page by page).
      - Splits text into manageable chunks for embedding/vector storage.
      - Keeps track of which files have already been ingested (via a metadata file).
    """

    def __init__(self, pdf_dir: str = "data", meta_file: str = "ingested_files.json"):
        """
        Initialize the PDF ingestor.

        Args:
            pdf_dir (str): Directory where PDFs are stored.
            meta_file (str): JSON file that records which PDFs have already been ingested.

        Behavior:
            - Reads the metadata file if it exists and loads a set of ingested file names.
            - If no metadata file exists, starts with an empty set.
            - Prepares a text splitter to break documents into chunks for embedding.
        """
        self.pdf_dir = pdf_dir
        self.meta_file = meta_file
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.ingested_files = set(json.load(f))
        else:
            self.ingested_files = set()

        # Recursive splitter ensures natural breakpoints (paragraphs, sentences).
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # max characters per chunk
            chunk_overlap=200  # overlap ensures context continuity
        )


    def _save_ingested(self):
        """
        Save the list of ingested files into the metadata JSON file.
        Ensures we don't repeatedly process the same PDFs.
        """
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self.ingested_files)), f, ensure_ascii=False, indent=2)



    def _load_new_pdfs(self) -> Tuple[List, List[str]]:
        """
        Load new PDFs that haven’t been ingested yet.

        Returns:
            Tuple[List, List[str]]:
                - A list of LangChain Documents (text + metadata).
                - A list of filenames of the new PDFs found.

        Notes:
            - Each document's metadata includes the "source" (file name).
            - Skips files that were already recorded in ingested_files.
        """
        docs, new_files = [], []
        if not os.path.isdir(self.pdf_dir):
            print(f"[ingest] PDF directory '{self.pdf_dir}' not found.")
            return docs, new_files

        for fname in os.listdir(self.pdf_dir):
            if fname.lower().endswith(".pdf") and fname not in self.ingested_files:
                path = os.path.join(self.pdf_dir, fname)
                try:
                    # Use LangChain’s PyPDFLoader to load pages from the PDF
                    loader = PyPDFLoader(path)
                    file_docs = loader.load()

                    # Attach metadata to each page
                    for d in file_docs:
                        d.metadata = d.metadata or {}
                        d.metadata["source"] = fname

                    docs.extend(file_docs)
                    new_files.append(fname)
                except Exception as e:
                    print(f"[ingest] Failed to load {fname}: {e}")
        return docs, new_files


    def ingest_new(self):
        """
        Ingest new PDFs into the system.

        Steps:
            1. Look for PDFs in the directory that haven't been ingested yet.
            2. Load their pages as documents.
            3. Split documents into smaller chunks (for embeddings).
            4. Update metadata file to mark these PDFs as processed.

        Returns:
            List: A list of document chunks ready for embedding.
        """
        print("[ingest] Checking for new PDFs...")
        docs, new_files = self._load_new_pdfs()

        if not docs:
            print("[ingest] No new PDFs to ingest.")
            return []

        print(f"[ingest] Found {len(new_files)} new PDFs, total pages: {len(docs)}")

        # Split each page into smaller overlapping chunks
        chunks = self.splitter.split_documents(docs)
        print(f"[ingest] Split into {len(chunks)} chunks.")

        # Update ingested metadata
        self.ingested_files.update(new_files)
        self._save_ingested()

        return chunks
