"""
LangGraph state definition.
This TypedDict flows through every node in the crawl graph.
Each node reads what it needs and writes what it produces.
"""
from typing import TypedDict, Optional, Any
from pydantic import BaseModel


class CrawlerState(TypedDict):
    # ─── Project config (set once at init, read-only during crawl) ───
    project_id: int
    mode: str                          # "TEXT_ONLY" or "SINGLE_IMAGE"
    target_query: str
    max_depth: int
    schema_class: Any                  # Pydantic model class (not serializable, passed by ref)

    # ─── Current page state (updated per URL) ───
    current_url: str
    current_depth: int
    navigation_ok: bool

    # ─── Page content (set by parse node) ───
    raw_html: str
    viewport_text: str                 # Viewport-only text with table context
    full_page_text: str                # Full page text for extraction
    screenshot_bytes: bytes
    table_contexts: list               # Detected table headers
    page_links: list                   # Links with context from DOM parser

    # ─── LLM decisions ───
    page_has_data: bool                # Triage result
    extracted_items: list              # Raw LLM output (Pydantic instances)
    validated_items: list              # After validator pass
    unique_items: list                 # After dedup pass
    scored_links: list                 # Links with confidence scores

    # ─── Crawl control ───
    should_continue: bool              # Master switch
    stall_count: int                   # Consecutive pages with no progress
    total_extracted: int               # Running total of unique items
    pages_processed: int
    error: Optional[str]               # Last error message

    # ─── Events for WebSocket streaming ───
    log_messages: list                 # Accumulated log messages this cycle
