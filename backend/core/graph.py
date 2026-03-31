"""
LangGraph graph definition — the crawl state machine.

Flow:
  pick_url → navigate → parse → classify
    ├─ (has_data) → extract → validate → dedupe → emit → steer_links → check_continue → pick_url
    └─ (no_data) → steer_links → check_continue → pick_url

Key design decision: steer_links ALWAYS runs (even after extraction)
to catch pagination links on product listing pages.
"""
from langgraph.graph import StateGraph, END
from .state import CrawlerState
from . import nodes


def build_crawl_graph() -> StateGraph:
    """Construct the LangGraph state machine for crawling."""

    graph = StateGraph(CrawlerState)

    # ─── Add nodes ────────────────────────────────────────────────
    graph.add_node("pick_url", nodes.pick_url)
    graph.add_node("navigate", nodes.navigate)
    graph.add_node("parse", nodes.parse)
    graph.add_node("classify", nodes.classify)
    graph.add_node("extract", nodes.extract)
    graph.add_node("validate", nodes.validate)
    graph.add_node("dedupe", nodes.dedupe)
    graph.add_node("emit", nodes.emit)
    graph.add_node("steer_links", nodes.steer_links)
    graph.add_node("check_continue", nodes.check_continue)

    # ─── Set entry point ──────────────────────────────────────────
    graph.set_entry_point("pick_url")

    # ─── Edges: pick_url ──────────────────────────────────────────
    def after_pick_url(state: CrawlerState) -> str:
        if not state.get("should_continue", True):
            return "end"
        if not state.get("current_url"):
            return "pick_url"  # Depth skip — try again
        return "navigate"

    graph.add_conditional_edges("pick_url", after_pick_url, {
        "navigate": "navigate",
        "pick_url": "pick_url",
        "end": END,
    })

    # ─── Edges: navigate ──────────────────────────────────────────
    def after_navigate(state: CrawlerState) -> str:
        if state.get("navigation_ok"):
            return "parse"
        return "check_continue"  # Skip to next URL

    graph.add_conditional_edges("navigate", after_navigate, {
        "parse": "parse",
        "check_continue": "check_continue",
    })

    # ─── Edges: parse → classify OR skip (if blocked/captcha) ─────
    def after_parse(state: CrawlerState) -> str:
        # If parse detected captcha/block/empty — skip classify entirely
        vt = state.get("viewport_text", "")
        if not vt or len(vt.strip()) < 50:
            return "check_continue"
        return "classify"

    graph.add_conditional_edges("parse", after_parse, {
        "classify": "classify",
        "check_continue": "check_continue",
    })

    # ─── Edges: classify ──────────────────────────────────────────
    def after_classify(state: CrawlerState) -> str:
        if state.get("page_has_data"):
            return "extract"
        return "steer_links"  # No data → find better links

    graph.add_conditional_edges("classify", after_classify, {
        "extract": "extract",
        "steer_links": "steer_links",
    })

    # ─── Edges: extraction pipeline ───────────────────────────────
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "dedupe")
    graph.add_edge("dedupe", "emit")
    # After emit, ALSO run link steering (fixes pagination bug)
    graph.add_edge("emit", "steer_links")

    # ─── Edges: steer_links → check_continue ─────────────────────
    graph.add_edge("steer_links", "check_continue")

    # ─── Edges: check_continue ────────────────────────────────────
    def after_check(state: CrawlerState) -> str:
        if state.get("should_continue", False):
            return "pick_url"
        return "end"

    graph.add_conditional_edges("check_continue", after_check, {
        "pick_url": "pick_url",
        "end": END,
    })

    return graph


def compile_crawl_graph():
    """Build and compile the graph for execution."""
    graph = build_crawl_graph()
    return graph.compile()
