import json
import sys
import os

sys.path.insert(0, "/content/multimodal-agentic-pipeline")


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
    query = state.get("perception_output") or state["query"]
    print(f"  [retrieval] searching: '{str(query)[:50]}'")
    stub_docs = [
        {"rank": 1, "score": 0.91, "chunk": "Consolidation indicates airspace filling, commonly pneumonia.", "source": "radiology_textbook.pdf"},
        {"rank": 2, "score": 0.84, "chunk": "Right lower lobe consolidation is most common in community-acquired pneumonia.", "source": "clinical_guidelines.pdf"},
        {"rank": 3, "score": 0.79, "chunk": "Bacterial pneumonia presents as lobar consolidation with air bronchograms.", "source": "radiology_textbook.pdf"}
    ]
    print(f"  [retrieval] {len(stub_docs)} docs retrieved")
    return {"retrieval_output": json.dumps(stub_docs)}


def output_node(state: dict) -> dict:
    print(f"  [output] validating and grounding...")

    perception_raw = state.get("perception_output", "")
    retrieval_raw = state.get("retrieval_output", "[]")

    # Handle empty perception — text-only queries skip perception node
    if perception_raw:
        try:
            perception = json.loads(perception_raw)
        except json.JSONDecodeError:
            perception = {"raw": perception_raw}
    else:
        perception = {}

    try:
        retrieval = json.loads(retrieval_raw)
    except json.JSONDecodeError:
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

    print(f"  [output] grounded={final['grounded']} citations={final['citation_count']} issues={len(issues)}")
    return {"final_answer": json.dumps(final, indent=2), "consistency_issues": issues}


def fail_node(state: dict) -> dict:
    print(f"  [fail] pipeline failed: {state.get('error')}")
    return {"final_answer": json.dumps({"error": state.get("error", "unknown"), "grounded": False})}
