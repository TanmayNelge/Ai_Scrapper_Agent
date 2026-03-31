"""
Builds runtime Pydantic models from user-defined schema dictionaries.
Supports nested types and optional fields.
"""
from pydantic import create_model, Field
from typing import Any, Dict, Optional, Type


TYPE_MAP = {
    "string": str,
    "str": str,
    "text": str,
    "number": float,
    "float": float,
    "integer": int,
    "int": int,
    "boolean": bool,
    "bool": bool,
    "list": list[str],
    "array": list[str],
    "url": str,
    "image_url": str,
}


def build_dynamic_model(model_name: str, schema_def: Dict[str, str]) -> Type:
    """
    Convert a simple schema dict into a Pydantic model.

    Example:
        schema_def = {"product_name": "string", "price_usd": "number", "in_stock": "boolean"}
        returns Pydantic model with typed fields + descriptions.
    """
    fields = {}
    for field_name, field_type_str in schema_def.items():
        raw = str(field_type_str).lower().strip()
        py_type = TYPE_MAP.get(raw, str)

        # Make all fields Optional with None default to reduce LLM failures
        fields[field_name] = (
            Optional[py_type],
            Field(default=None, description=f"The extracted {field_name}"),
        )

    return create_model(model_name, **fields)


def schema_fields_description(schema_class: Type) -> str:
    """Human-readable field list for LLM prompts."""
    props = schema_class.model_json_schema().get("properties", {})
    parts = []
    for name, info in props.items():
        # Get inner type from anyOf (Optional creates anyOf)
        type_info = info.get("type", "string")
        if "anyOf" in info:
            types = [t.get("type", "") for t in info["anyOf"] if t.get("type") != "null"]
            type_info = types[0] if types else "string"
        parts.append(f"'{name}' ({type_info})")
    return ", ".join(parts)
