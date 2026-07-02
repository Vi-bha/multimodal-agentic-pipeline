import json
import sys
import os

sys.path.insert(0, "/content/multimodal-agentic-pipeline")

# ── Real RAGRetriever (replaces stub) ────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional


class RAGRetriever:
    def __init__(self, db_path="data/chromadb", collection_name="medical_knowledge"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(6, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            candidates.append({
                "chunk": doc,
                "source": meta["source"],
                "topic": meta["topic"],
                "initial_score": round(1 - dist, 4)
            })

        pairs = [[query, c["chunk"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for i, s in enumerate(scores):
            candidates[i]["rerank_score"] = round(float(s), 4)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        for i, r in enumerate(reranked):
            r["rank"] = i + 1
        return reranked


# ── Lazy-loaded retriever singleton ─────────────────────────
_retriever = None

def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


# ── Node functions ───────────────────────────────────────────
def perception_node(state: dict) -> dict:
    query = state["query"]
    print(f"  [perception] query='{query[:50]}'")
    stub_output = json.dumps({
        "findings": [
            {"finding": "consolidation", "present": True, "confidence": "high", "location": "right lower lobe"},
            {"finding": "pleural effusion", "present": False, "confidence": "medium", "location": None},
            {"finding": "cardiomegaly", "present": False, "confidence": "high", "location": None}
        ],
        "overall_impression": "Focal consolidation in right lower lobe consistent with pneumonia."
    })
    print(f"  [perception] findings generated")
    return {"perception_output": stub_output, "error": None, "retry_count": 0}


def retrieval_node(state: dict) -> dict:
    """
    REAL retrieval — queries ChromaDB + CrossEncoder reranker.
    Uses perception output as query if available, else raw user query.
    """
    # Build query from perception output or raw query
    perception_raw = state.get("perception_output", "")
    if perception_raw:
        try:
            perception = json.loads(perception_raw)
            # Extract present findings as search query
            present = [f["finding"] for f in perception.get("findings", []) if f["present"]]
            query = " ".join(present) if present else state["query"]
        except:
            query = state["query"]
    else:
        query = state["query"]

    print(f"  [retrieval] searching: '{query[:60]}'")

    retriever = get_retriever()
    docs = retriever.retrieve(query, top_k=3)

    if not docs:
        print(f"  [retrieval] ⚠️ no docs found — ChromaDB may be empty")
        return {"retrieval_output": json.dumps([])}

    print(f"  [retrieval] ✅ {len(docs)} docs retrieved")
    return {"retrieval_output": json.dumps(docs)}


def output_node(state: dict) -> dict:
    print(f"  [output] validating and grounding...")

    perception_raw = state.get("perception_output", "")
    retrieval_raw = state.get("retrieval_output", "[]")

    perception = json.loads(perception_raw) if perception_raw else {}
    try:
        retrieval = json.loads(retrieval_raw)
    except:
        retrieval = []

    issues = []
    if "findings" in perception and "overall_impression" in perception:
        impression = perception["overall_impression"].lower()
        for f in perception["findings"]:
            name = f["finding"].lower()
            if f["present"] and name not in impression:
                issues.append(f"Finding {name} present but missing from impression")

    final = {
        "perception": perception,
        "retrieved_evidence": retrieval,
        "consistency_issues": issues,
        "grounded": len(issues) == 0,
        "citation_count": len(retrieval)
    }

    print(f"  [output] ✅ grounded={final['grounded']} citations={final['citation_count']} issues={len(issues)}")
    return {"final_answer": json.dumps(final, indent=2), "consistency_issues": issues}


def fail_node(state: dict) -> dict:
    print(f"  [fail] pipeline failed: {state.get('error')}")
    return {"final_answer": json.dumps({"error": state.get("error", "unknown"), "grounded": False})}
