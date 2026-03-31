"""
URL Frontier replaces the simple DFS stack + visited_links set.
Uses a priority queue (highest confidence first) and a bloom-filter-like
set for O(1) visited checks with URL normalization.
"""
import heapq
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode
from typing import Optional
from dataclasses import dataclass, field


# Tracking params to strip for URL normalization
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid", "click_id",
    "srsltid", "scid", "aff_id", "tracking", "_ga", "trk",
}


def normalize_url(url: str, base_url: str = None) -> Optional[str]:
    """
    Normalize a URL by:
    1. Resolving relative paths against base_url
    2. Stripping tracking parameters
    3. Removing fragments
    4. Lowercasing scheme + host
    5. Removing trailing slashes
    6. Rejecting tracking/ad URLs
    Returns None for invalid/non-HTTP URLs or junk tracking links.
    """
    if not url:
        return None

    # Skip non-HTTP
    if url.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
        return None

    # REJECT: URLs that are obviously tracking/ad redirects
    # These waste time and lead to empty/redirect pages
    lower = url.lower()
    junk_patterns = [
        "/sp/track?", "/click?", "/redirect?", "/redir?",
        "doubleclick.net", "googlesyndication", "adsystem",
        "clickserve", "clickthrough", "adclick",
        "beacon.", "pixel.", "analytics.",
        "eventST=click",  # Walmart ad tracker (from your logs)
    ]
    for junk in junk_patterns:
        if junk.lower() in lower:
            return None

    # REJECT: Absurdly long URLs (almost always tracking garbage)
    if len(url) > 500:
        return None

    # Resolve relative URLs
    if base_url and not url.startswith(("http://", "https://")):
        url = urljoin(base_url, url)

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    if parsed.scheme not in ("http", "https"):
        return None

    # Strip tracking params
    query_params = parse_qs(parsed.query, keep_blank_values=False)
    clean_params = {
        k: v for k, v in query_params.items()
        if k.lower() not in TRACKING_PARAMS
    }
    clean_query = urlencode(clean_params, doseq=True) if clean_params else ""

    # Rebuild normalized
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        parsed.params,
        clean_query,
        "",  # Drop fragment
    ))
    return normalized


@dataclass(order=True)
class ScoredURL:
    """Priority queue entry. Lower score = higher priority (heapq is min-heap)."""
    priority: float  # negative confidence so higher confidence pops first
    url: str = field(compare=False)
    depth: int = field(compare=False)
    source_url: str = field(compare=False, default="")


class URLFrontier:
    """
    Priority queue for URLs scored by LLM confidence.
    Bloom filter (hash set) prevents revisiting.
    """

    def __init__(self):
        self._queue: list[ScoredURL] = []
        self._visited: set[str] = set()
        self._enqueued: set[str] = set()  # Prevent duplicate queue entries

    def add(self, url: str, depth: int, confidence: float = 0.5,
            source_url: str = "", base_url: str = None):
        """
        Add a URL to the frontier if not already visited or enqueued.
        confidence: 0.0 (unlikely useful) to 1.0 (definitely has target data)
        """
        normalized = normalize_url(url, base_url)
        if normalized is None:
            return False

        if normalized in self._visited or normalized in self._enqueued:
            return False

        entry = ScoredURL(
            priority=-confidence,  # Negate for min-heap
            url=normalized,
            depth=depth,
            source_url=source_url,
        )
        heapq.heappush(self._queue, entry)
        self._enqueued.add(normalized)
        return True

    def add_seeds(self, urls: list[str], depth: int = 1):
        """Bulk-add seed URLs with default confidence."""
        added = 0
        for url in urls:
            if self.add(url, depth, confidence=0.5):
                added += 1
        return added

    def pop(self) -> Optional[ScoredURL]:
        """Pop the highest-confidence URL from the frontier."""
        while self._queue:
            entry = heapq.heappop(self._queue)
            self._enqueued.discard(entry.url)

            if entry.url in self._visited:
                continue

            return entry
        return None

    def mark_visited(self, url: str):
        """Mark a URL as visited so it won't be re-crawled."""
        normalized = normalize_url(url) or url
        self._visited.add(normalized)

    def is_visited(self, url: str) -> bool:
        normalized = normalize_url(url) or url
        return normalized in self._visited

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def visited_count(self) -> int:
        return len(self._visited)

    @property
    def is_empty(self) -> bool:
        # Check if all remaining queue entries are already visited
        return all(e.url in self._visited for e in self._queue) and not self._queue

    def clear_queue(self):
        """Clear the queue but keep visited set (for branch abandonment)."""
        self._queue.clear()
        self._enqueued.clear()
