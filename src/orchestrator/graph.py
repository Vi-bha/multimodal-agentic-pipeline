from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional
import json


class AgentState(TypedDict):
    """
    Shared state object flowing through every node.
    Each node reads from and writes back to this state.
    """
    query: str
    image_provided: bool
    perception_output: str
    retrieval_output: str
    final_answer: str
    route: str
    consistency_issues: list
    error: Optional[str]
    retry_count: int


def validate_state(state: AgentState, required_keys: list) -> Optional[str]:
    """Validate state has required non-empty keys before node runs."""
    for key in required_keys:
        if key not in state or state[key] is None:
            return f"Missing required key: '{key}'"
        if isinstance(state[key], str) and not state[key].strip():
            return f"Empty value for key: '{key}'"
    return None


def router_node(state: AgentState) -> AgentState:
    error = validate_state(state, ["query"])
    if error:
        return {"error": error, "route": ""}
    route = "perception" if state["image_provided"] else "retrieval"
    return {"route": route, "error": None}


def route_decision(state: AgentState) -> Literal["perception", "retrieval"]:
    return state["route"]


def should_retry(state: AgentState) -> Literal["retry_perception", "retrieval", "fail"]:
    """
    After perception node — decide next step based on error state.
    Retries up to 2 times before failing gracefully.
    """
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry_perception"
    elif state.get("error") and state.get("retry_count", 0) >= 2:
        return "fail"
    return "retrieval"


def fail_node(state: AgentState) -> AgentState:
    return {"final_answer": json.dumps({
        "error": state.get("error"),
        "grounded": False
    })}


def build_graph() -> StateGraph:
    """
    Build production graph with:
    - Conditional routing (image vs text-only)
    - Retry logic on perception node (handles OOM, timeout)
    - Graceful failure path
    - Stub nodes replaced on Day 15
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("perception", lambda s: {"perception_output": "[VLM stub]", "error": None, "retry_count": 0})
    graph.add_node("retrieval", lambda s: {"retrieval_output": "[RAG stub]"})
    graph.add_node("output", lambda s: {"final_answer": json.dumps({
        "perception": s.get("perception_output", ""),
        "evidence": s.get("retrieval_output", ""),
        "grounded": True
    })})
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision,
        {"perception": "perception", "retrieval": "retrieval"})
    graph.add_conditional_edges("perception", should_retry, {
        "retry_perception": "perception",
        "retrieval": "retrieval",
        "fail": "fail"
    })
    graph.add_edge("retrieval", "output")
    graph.add_edge("output", END)
    graph.add_edge("fail", END)

    return graph.compile()


def run_pipeline(query: str, image_provided: bool = False) -> dict:
    """Main entry point for the pipeline."""
    app = build_graph()
    result = app.invoke({
        "query": query,
        "image_provided": image_provided,
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
    app = build_graph()
    print("Pipeline ready")
    result = run_pipeline("What findings are visible?", image_provided=True)
    print(f"Result: {result}")
