"""
Main entry point for the Multimodal Agentic Pipeline.
Run: python src/orchestrator/main.py
FastAPI wrapper added Day 15.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional
from src.orchestrator.nodes import perception_node, retrieval_node, output_node, fail_node


class AgentState(TypedDict):
    query: str
    image_provided: bool
    image_path: Optional[str]
    perception_output: str
    retrieval_output: str
    final_answer: str
    route: str
    consistency_issues: list
    error: Optional[str]
    retry_count: int


def router_node(state: AgentState) -> AgentState:
    if not state.get("query", "").strip():
        return {"error": "Empty query", "route": ""}
    route = "perception" if state["image_provided"] else "retrieval"
    print(f"  [router] -> {route}")
    return {"route": route, "error": None}


def route_decision(state: AgentState) -> Literal["perception", "retrieval"]:
    return state["route"]


def should_retry(state: AgentState) -> Literal["retry_perception", "retrieval", "fail"]:
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry_perception"
    elif state.get("error") and state.get("retry_count", 0) >= 2:
        return "fail"
    return "retrieval"


def build_pipeline():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("perception", perception_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("output", output_node)
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision,
        {"perception": "perception", "retrieval": "retrieval"})
    graph.add_conditional_edges("perception", should_retry,
        {"retry_perception": "perception", "retrieval": "retrieval", "fail": "fail"})
    graph.add_edge("retrieval", "output")
    graph.add_edge("output", END)
    graph.add_edge("fail", END)

    return graph.compile()


def run(query: str, image_provided: bool = False, image_path: str = None) -> dict:
    app = build_pipeline()
    print("\n" + "="*60)
    print(f"PIPELINE RUN | query: {query[:50]} | image: {image_provided}")
    print("="*60)

    result = app.invoke({
        "query": query,
        "image_provided": image_provided,
        "image_path": image_path,
        "perception_output": "",
        "retrieval_output": "",
        "final_answer": "",
        "route": "",
        "consistency_issues": [],
        "error": None,
        "retry_count": 0
    })
    return json.loads(result["final_answer"])


if __name__ == "__main__":
    result1 = run("What findings are visible in this chest X-ray?", image_provided=True)
    print("\n--- RESULT 1 ---")
    print(json.dumps(result1, indent=2))

    result2 = run("What is consolidation in chest X-ray?", image_provided=False)
    print("\n--- RESULT 2 ---")
    print(json.dumps(result2, indent=2))
