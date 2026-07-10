from __future__ import annotations

from copy import deepcopy
from typing import Any


def restricted_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Compile Pydantic schema into the conservative cross-provider subset used by v0."""

    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("default", "examples", "deprecated", "readOnly", "writeOnly"):
                node.pop(key, None)
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
                node["additionalProperties"] = False
            if node.get("type") == "object" and "properties" not in node:
                node.setdefault("properties", {})
                node["additionalProperties"] = False
            for child in list(node.values()):
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(result)
    return result


def schema_name(task_id: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in task_id)
    return f"video_generator_{cleaned}_v1"[:64]
