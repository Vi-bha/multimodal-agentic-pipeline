import json
import re
from typing import Dict, List, Any


def parse_vlm_json(raw_output: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from VLM output.
    Handles markdown code fences and other common formatting artifacts.
    """
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
    cleaned = re.sub(r"^```\s*|\s*```$", "", cleaned.strip())

    try:
        parsed = json.loads(cleaned)
        return {"success": True, "data": parsed, "error": None}
    except json.JSONDecodeError as e:
        return {"success": False, "data": None, "error": str(e)}


def check_consistency(parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if structured findings are consistent with the impression text.
    Handles negation phrases like "no X" and "without X" correctly.
    """
    impression = parsed_data["overall_impression"].lower()
    issues = []

    for f in parsed_data["findings"]:
        finding_name = f["finding"].split("/")[0].lower().strip()

        negated_pattern = rf"(no|without|absence of)\s+{re.escape(finding_name)}"
        is_negated_in_text = bool(re.search(negated_pattern, impression))
        is_mentioned = finding_name in impression

        if is_negated_in_text:
            impression_claims_present = False
        elif is_mentioned:
            impression_claims_present = True
        else:
            impression_claims_present = None

        if impression_claims_present is not None and impression_claims_present != f["present"]:
            issues.append({
                "finding": f["finding"],
                "structured_says": f["present"],
                "impression_implies": impression_claims_present
            })

    return issues


class StructuredOutputNode:
    """
    Validates and grounds VLM output.
    Third node in the LangGraph pipeline — parses VLM JSON output
    and flags inconsistencies between structured findings and summary.
    """

    def process(self, raw_vlm_output: str) -> Dict[str, Any]:
        parse_result = parse_vlm_json(raw_vlm_output)

        if not parse_result["success"]:
            return {
                "valid": False,
                "error": parse_result["error"],
                "data": None,
                "consistency_issues": []
            }

        issues = check_consistency(parse_result["data"])

        return {
            "valid": True,
            "error": None,
            "data": parse_result["data"],
            "consistency_issues": issues,
            "is_grounded": len(issues) == 0
        }
