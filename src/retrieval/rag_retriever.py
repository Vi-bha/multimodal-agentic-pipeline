import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional


class RAGRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: ChromaDB vector search (fast, approximate)
    Stage 2: CrossEncoder reranking (accurate)

    This is the retrieval node in the LangGraph pipeline.
    Replaces stub retrieval_node on Day 15.
    """

    def __init__(
        self,
        db_path: str = "data/chromadb",
        collection_name: str = "medical_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("[RAGRetriever] Initializing...")

        # ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Embedder
        self.embedder = SentenceTransformer(embedding_model)

        # Reranker
        self.reranker = CrossEncoder(reranker_model)

        print(f"[RAGRetriever] Ready — {self.collection.count()} chunks indexed")

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to ChromaDB.
        Each document: {"text": str, "source": str, "topic": str}
        """
        texts = [d["text"] for d in documents]
        embeddings = self.embedder.encode(texts).tolist()
        start_id = self.collection.count()

        self.collection.add(
            ids=[f"doc_{start_id + i}" for i in range(len(documents))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": d["source"], "topic": d.get("topic", "general")} for d in documents]
        )
        print(f"[RAGRetriever] Added {len(documents)} documents. Total: {self.collection.count()}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 6,
        topic_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Two-stage retrieval: ChromaDB → CrossEncoder reranker.
        Returns top_k most relevant chunks with scores and citations.
        """
        if self.collection.count() == 0:
            return []

        # Stage 1: ChromaDB vector search
        query_embedding = self.embedder.encode([query]).tolist()
        where = {"topic": topic_filter} if topic_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(candidate_k, self.collection.count()),
            where=where,
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

        # Stage 2: CrossEncoder reranking
        pairs = [[query, c["chunk"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = round(float(score), 4)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        for i, r in enumerate(reranked):
            r["rank"] = i + 1

        return reranked
