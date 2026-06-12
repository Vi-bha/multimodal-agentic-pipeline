from sentence_transformers import SentenceTransformer, util
import json
import sys

CORPUS = [
    "Cardiomegaly is an enlargement of the heart visible on chest X-ray.",
    "Pleural effusion refers to fluid accumulation in the pleural space.",
    "Pneumonia presents as consolidation or ground-glass opacity in lung fields.",
    "Atelectasis is partial or complete lung collapse seen as increased opacity.",
    "No acute cardiopulmonary findings. Heart size within normal limits.",
    "Bilateral hilar lymphadenopathy may suggest sarcoidosis or lymphoma.",
    "Pneumothorax is air in the pleural cavity causing lung collapse.",
    "Aortic knuckle is prominent suggesting possible aortic aneurysm.",
]

def run_similarity_search(query: str, top_k: int = 3) -> dict:
    print(f"[INFO] Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[INFO] Encoding corpus ({len(CORPUS)} chunks)...")
    corpus_embeddings = model.encode(CORPUS, convert_to_tensor=True)

    print(f"[INFO] Encoding query: {query}")
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    top_results = sorted(
        enumerate(scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = {
        "query": query,
        "top_k": top_k,
        "results": [
            {
                "rank": i + 1,
                "score": round(score, 4),
                "chunk": CORPUS[idx],
            }
            for i, (idx, score) in enumerate(top_results)
        ]
    }
    return results

if __name__ == "__main__":
    query = "fluid in the chest"
    results = run_similarity_search(query)

    print("\n" + "="*60)
    print(f"QUERY: {results['query']}")
    print("="*60)
    for r in results["results"]:
        print(f"\nRank {r['rank']} | Score: {r['score']}")
        print(f"  -> {r['chunk']}")
    print("="*60)

    with open("docker/day01_practice/output_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[INFO] Results saved to output_results.json")
