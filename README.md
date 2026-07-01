# multimodal-agentic-pipeline

> A multimodal agentic system where a LangGraph state machine orchestrates vision, retrieval, and structured output — grounded answers, zero hallucination.

![Status](https://img.shields.io/badge/status-building-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What This Is

A **domain-agnostic agentic pipeline** that takes an image + text query and returns a grounded, cited, structured answer.

The domain used for development is medical imaging — specifically because hallucination cost is high, which makes grounding quality measurable. The architecture itself is fully domain-agnostic.

```
Input: image + text query
         │
         ▼
┌─────────────────────────────────────────────┐
│           LangGraph Orchestrator            │
│                                             │
│  ┌──────────────┐     ┌──────────────────┐  │
│  │  VLM Node    │────▶│  RAG Retrieval   │  │
│  │  Qwen2-VL    │     │  ChromaDB +      │  │
│  │  perception  │     │  Reranker        │  │
│  └──────────────┘     └──────────────────┘  │
│                              │               │
│                              ▼               │
│                   ┌──────────────────┐       │
│                   │  Structured      │       │
│                   │  Output Node     │       │
│                   │  JSON Schema     │       │
│                   └──────────────────┘       │
└─────────────────────────────────────────────┘
         │
         ▼
Output: grounded, cited, structured JSON answer
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| VLM Inference | Qwen2-VL |
| Image Embeddings | CLIP / ViT |
| Agent Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Reranking | CrossEncoder |
| Backend | FastAPI |
| Containerisation | Docker |
| Demo UI | Gradio (HuggingFace Spaces) |
| Model Hosting | HuggingFace Hub |

---

## Architecture

### Agent Flow

```
User Query (image + text)
        │
        ▼
   [Router Node] ──────────────────────────────────┐
        │                                           │
        ▼                                           ▼
[VLM Perception Node]                   [RAG Retrieval Node]
  Qwen2-VL extracts                     ChromaDB similarity
  visual findings                       search + reranker
        │                                           │
        └──────────────┬────────────────────────────┘
                       ▼
            [Structured Output Node]
              JSON schema validation
              citation linking
                       │
                       ▼
              Final Grounded Answer
```

### Key Design Decisions

- **LangGraph over LangChain**: explicit state machine gives deterministic routing and debuggable agent traces
- **Qwen2-VL over GPT-4V**: open-weight, self-hostable, no API cost
- **ChromaDB over Pinecone**: local-first, Docker-friendly, no external dependency
- **Medical domain for dev**: high hallucination cost = measurable grounding quality

---

## Project Structure

```
multimodal-agentic-pipeline/
├── src/
│   ├── orchestrator/        # LangGraph state machine
│   ├── perception/          # Qwen2-VL VLM tool
│   ├── retrieval/           # ChromaDB RAG pipeline
│   └── output/              # JSON schema validation
├── docker/
│   ├── day01_practice/      # Docker fundamentals exercise
│   └── Dockerfile           # Production Dockerfile
├── data/
│   ├── raw/                 # Raw images + reports
│   └── processed/           # Embeddings, chunks
├── notebooks/               # Experiments and exploration
├── tests/                   # Unit + integration tests
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Build Log

Tracking progress day by day.

| Day | Focus | Status |
|---|---|---|
| 1–3 | Docker fundamentals — containerise Python app | ✅ Day 1 Done |
| 4–7 | CLIP + ViT embeddings, Qwen2-VL inference | ✅ Done |
| 8–10 | LangGraph fundamentals — minimal 2-tool agent | 🔄 Up Next |
| 11–14 | RAG pipeline — ChromaDB + reranker | ⏳ Pending |
| 15–18 | Full pipeline integration + FastAPI + Docker | ⏳ Pending |
| 19–22 | HuggingFace Spaces deployment + Gradio UI | ⏳ Pending |
| 23–30 | Job applications + interview prep | ⏳ Pending |

---

## Demo

> 🚧 Coming Day 19 — Gradio UI on HuggingFace Spaces

---

## Running Locally

```bash
# Clone
git clone https://github.com/Vi-bha/multimodal-agentic-pipeline.git
cd multimodal-agentic-pipeline

# Run with Docker
docker-compose up --build

# Or run directly
pip install -r requirements.txt
python src/orchestrator/main.py
```

---

## About

Built as part of a 30-day sprint toward an Agentic AI Engineer role.

**Research background:** M.Tech, MANIT Bhopal | 2 published papers (Springer + Scopus)

**Target identity:** Agentic AI Engineer · LLM Systems Engineer · AI Research Engineer
