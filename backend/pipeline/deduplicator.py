"""
Two-stage deduplicator: SimHash (coarse) → RapidFuzz (fine).
Fix: stores Simhash objects directly instead of rebuilding per comparison.
"""
from simhash import Simhash
from rapidfuzz import fuzz
from typing import Any
from ..config import CONFIG


class Deduplicator:
    def __init__(self):
        cfg = CONFIG.dedup
        self.simhash_threshold = cfg.simhash_threshold
        self.fuzz_threshold = cfg.fuzz_threshold

        self._signatures: list[Simhash] = []
        self._texts: list[str] = []
        self._name_keys: set[str] = set()  # Stage 0: exact name dedup

    def _flatten(self, data: Any) -> str:
        """Flatten dict/model values into a single comparison string."""
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if not isinstance(data, dict):
            return str(data).lower()

        parts = []
        for v in data.values():
            if isinstance(v, str) and v.strip():
                parts.append(v)
            elif isinstance(v, (int, float)):
                parts.append(str(v))
            elif isinstance(v, list):
                parts.extend(str(item) for item in v if item)
        return " ".join(parts).lower()

    def _get_name_key(self, data: Any) -> str:
        """
        Extract the first meaningful string field as a 'name key'.
        Catches: same product scraped from different pages with different prices.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if not isinstance(data, dict):
            return ""

        for key, val in data.items():
            if isinstance(val, str) and len(val.strip()) > 3:
                # Normalize: lowercase, strip whitespace, remove extra spaces
                clean = " ".join(val.lower().split())
                return clean
        return ""

    def is_duplicate(self, new_data: Any) -> bool:
        """
        Three-stage dedup:
        Stage 0: Exact name match (fast, catches same product different page)
        Stage 1: SimHash coarse filter
        Stage 2: RapidFuzz fine check
        """
        text = self._flatten(new_data)
        if not text.strip():
            return True

        # Stage 0: Name-based exact match
        name_key = self._get_name_key(new_data)
        if name_key and name_key in self._name_keys:
            return True

        new_sig = Simhash(text)

        # Stage 1: Find coarse candidates
        candidates = []
        for idx, stored_sig in enumerate(self._signatures):
            if new_sig.distance(stored_sig) <= self.simhash_threshold:
                candidates.append(idx)

        if not candidates:
            self._signatures.append(new_sig)
            self._texts.append(text)
            if name_key:
                self._name_keys.add(name_key)
            return False

        # Stage 2: Fine check with RapidFuzz
        for idx in candidates:
            similarity = fuzz.ratio(text, self._texts[idx])
            if similarity >= self.fuzz_threshold:
                return True

        self._signatures.append(new_sig)
        self._texts.append(text)
        if name_key:
            self._name_keys.add(name_key)
        return False

    @property
    def count(self) -> int:
        return len(self._signatures)

    def clear(self):
        self._signatures.clear()
        self._texts.clear()
