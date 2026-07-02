"""
Knowledge base ingestion pipeline.
Reads text files from data/processed/, chunks them, embeds, stores in ChromaDB.
Run: python src/retrieval/ingest.py
"""

import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

sys.path.insert(0, "/content/multimodal-agentic-pipeline")


def chunk_text(
    text: str,
    source: str,
    topic: str,
    chunk_size: int = 80,
    overlap: int = 20
) -> List[Dict]:
    """Split document into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "text": chunk,
            "source": source,
            "topic": topic,
            "chunk_index": len(chunks),
            "word_count": end - start
        })
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def detect_topic(text: str) -> str:
    """Simple keyword-based topic detection for chunks."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["pneumonia", "consolidation", "ground-glass"]):
        return "pneumonia"
    elif any(w in text_lower for w in ["effusion", "costophrenic", "transudate", "exudate"]):
        return "pleural_effusion"
    elif any(w in text_lower for w in ["cardiomegaly", "cardiothoracic", "cardiac silhouette"]):
        return "cardiomegaly"
    elif any(w in text_lower for w in ["pneumothorax", "pleural line", "tension"]):
        return "pneumothorax"
    elif any(w in text_lower for w in ["atelectasis", "volume loss", "plate-like"]):
        return "atelectasis"
    else:
        return "general"


def ingest_directory(
    data_dir: str = "data/processed",
    db_path: str = "data/chromadb",
    collection_name: str = "medical_knowledge",
    chunk_size: int = 80,
    overlap: int = 20,
    reset: bool = True
) -> int:
    """
    Ingest all .txt files from data_dir into ChromaDB.
    Returns total chunks stored.
    """
    # Init ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[ingest] Cleared existing collection")
        except:
            pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Load embedder
    print("[ingest] Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Process each file
    all_chunks = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    print(f"[ingest] Found {len(txt_files)} files: {txt_files}")

    for filename in txt_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            text = f.read()

        # Chunk the document
        raw_chunks = chunk_text(
            text=text,
            source=filename,
            topic="auto",
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Detect topic per chunk
        for chunk in raw_chunks:
            chunk["topic"] = detect_topic(chunk["text"])

        all_chunks.extend(raw_chunks)
        print(f"[ingest] {filename} → {len(raw_chunks)} chunks")

    # Embed all chunks
    print(f"[ingest] Embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # Store in ChromaDB
    collection.add(
        ids=[f"doc_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source": c["source"],
            "topic": c["topic"],
            "chunk_index": c["chunk_index"]
        } for c in all_chunks]
    )

    total = collection.count()
    print(f"[ingest] Done — {total} chunks stored in ChromaDB")
    return total


if __name__ == "__main__":
    total = ingest_directory()
    print(f"\n✅ Ingestion complete: {total} chunks ready for retrieval")
