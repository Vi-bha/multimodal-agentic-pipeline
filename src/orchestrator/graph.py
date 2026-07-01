from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional


class AgentState(TypedDict):
    """
    The state object that flows through every node in the pipeline.
    Each node reads from this state and writes back to it.
    """
    query: str
    image_provided: bool
    perception_output: str
    retrieval_output: str
    final_answer: str
    route: str
    consistency_issues: list


def router_node(state: AgentState) -> AgentState:
    """
    Entry point. Routes to perception if image provided,
    directly to retrieval for text-only queries.
    """
    route = "perception" if state["image_provided"] else "retrieval"
    print(f"[router] routing to -> {route}")
    return {"route": route}


def route_decision(state: AgentState) -> Literal["perception", "retrieval"]:
    """Conditional edge function — return value = next node name."""
    return state["route"]


def build_graph():
    """
    Build and compile the LangGraph state machine.
    Stub nodes replaced with real implementations on Day 15.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("perception", lambda s: {"perception_output": "[VLM stub]"})
    graph.add_node("retrieval", lambda s: {"retrieval_output": "[RAG stub]"})
    graph.add_node("output", lambda s: {"final_answer": f"perception={s.get('perception_output','')} retrieval={s.get('retrieval_output','')}"})

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"perception": "perception", "retrieval": "retrieval"}
    )
    graph.add_edge("perception", "retrieval")
    graph.add_edge("retrieval", "output")
    graph.add_edge("output", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print("Graph compiled successfully")
    result = app.invoke({
        "query": "What findings are visible in this chest X-ray?",
        "image_provided": True,
        "perception_output": "",
        "retrieval_output": "",
        "final_answer": "",
        "route": "",
        "consistency_issues": []
    })
    print(f"Final answer: {result['final_answer']}")
