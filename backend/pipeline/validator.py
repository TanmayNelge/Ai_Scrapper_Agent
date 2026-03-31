"""
Data Validator — catches junk extractions before they reach the deduplicator.
Checks that extracted items have meaningful content, not just nulls and zeros.
"""
from pydantic import BaseModel
from typing import Any


class DataValidator:
    """Validates extracted items have enough real data to be worth storing."""

    def __init__(self, min_filled_ratio: float = 0.3):
        """
        min_filled_ratio: at least this fraction of fields must be non-null/non-empty.
        0.3 means 30% of fields must have real data.
        """
        self.min_filled_ratio = min_filled_ratio

    def validate(self, item: Any) -> bool:
        """
        Returns True if the item has enough real data to keep.
        Rejects items that are mostly nulls, empty strings, or zeros.
        """
        if item is None:
            return False

        if isinstance(item, BaseModel):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = item
        else:
            return False

        if not data:
            return False

        total_fields = len(data)
        if total_fields == 0:
            return False

        filled = 0
        for value in data.values():
            if self._is_meaningful(value):
                filled += 1

        ratio = filled / total_fields
        return ratio >= self.min_filled_ratio

    def _is_meaningful(self, value: Any) -> bool:
        """Check if a value contains actual data."""
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip()
            # Reject empty, "N/A", "null", "none", single-char noise
            if not stripped or stripped.lower() in ("n/a", "null", "none", "-", "—", "na"):
                return False
            return len(stripped) > 1
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, bool):
            return True  # Booleans are always meaningful
        if isinstance(value, list):
            return len(value) > 0 and any(self._is_meaningful(v) for v in value)
        return True  # Unknown types pass

    def validate_batch(self, items: list[Any]) -> list[Any]:
        """Filter a list, keeping only valid items."""
        return [item for item in items if self.validate(item)]
