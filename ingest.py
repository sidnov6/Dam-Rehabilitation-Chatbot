"""
Ingest the dam rehabilitation PDF into ChromaDB for RAG retrieval.
Run this once before starting the chatbot: python3 ingest.py
"""

import os
import sys
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions

PDF_PATH = "Mnul_for_Rhbltn_of_Lrge_Dam.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "dam_rehab_manual"
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between chunks


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF."""
    pages = []
    print(f"Extracting text from: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{total} pages...")
    print(f"Extracted text from {len(pages)} pages.")
    return pages


def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split page texts into overlapping chunks."""
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "start_char": start,
                })
            start += chunk_size - overlap
    print(f"Created {len(chunks)} chunks.")
    return chunks


def ingest(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at '{pdf_path}'")
        sys.exit(1)

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Remove existing collection so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch upsert (ChromaDB handles up to 5461 per batch by default)
    batch_size = 500
    total = len(chunks)
    print(f"Embedding and storing {total} chunks in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"], "start_char": c["start_char"]} for c in batch],
        )
        print(f"  Stored chunks {i + 1}–{min(i + batch_size, total)} / {total}")

    print(f"\nIngestion complete. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")


if __name__ == "__main__":
    ingest(PDF_PATH)
