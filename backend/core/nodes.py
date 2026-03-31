"""
LangGraph nodes — each function is a graph node that processes CrawlerState.
Nodes are pure functions: read state → do work → return state updates.

Node map:
  pick_url → navigate → parse → classify → [extract | steer_links] → dedupe → emit
"""
import asyncio
from typing import Any
from ..browser.browser_manager import BrowserManager
from ..browser.dom_parser import DOMParser
from ..pipeline.classifier import PageClassifier
from ..pipeline.extractor import DataExtractor
from ..pipeline.validator import DataValidator
from ..pipeline.deduplicator import Deduplicator
from ..llm.llm_gateway import LLMGateway
from ..llm.binary_llm import BinaryLLM
from ..utils.url_frontier import URLFrontier
from ..utils.watchdog import Watchdog, run_with_timeout
from ..utils.event_bus import EventBus
from ..utils.image_processor import ImageProcessor
from ..config import CONFIG
from .state import CrawlerState


class NodeContext:
    """
    Shared mutable resources that nodes need.
    Not part of the state dict — these are long-lived objects.
    """
    def __init__(
        self,
        browser: BrowserManager,
        frontier: URLFrontier,
        watchdog: Watchdog,
        event_bus: EventBus,
        classifier: PageClassifier,
        extractor: DataExtractor,
        validator: DataValidator,
        deduplicator: Deduplicator,
        llm_gateway: LLMGateway,
        dom_parser: DOMParser,
    ):
        self.browser = browser
        self.frontier = frontier
        self.watchdog = watchdog
        self.event_bus = event_bus
        self.classifier = classifier
        self.extractor = extractor
        self.validator = validator
        self.deduplicator = deduplicator
        self.llm = llm_gateway
        self.parser = dom_parser


# We store the context in a module-level variable set by the orchestrator
_ctx: NodeContext = None


def set_node_context(ctx: NodeContext):
    global _ctx
    _ctx = ctx


# ─── Node: pick_url ──────────────────────────────────────────────
async def pick_url(state: CrawlerState) -> dict:
    """Pop the highest-priority URL from the frontier."""
    ctx = _ctx
    entry = ctx.frontier.pop()

    if entry is None:
        await ctx.event_bus.log("Frontier empty — no more URLs to process.")
        return {
            "should_continue": False,
            "log_messages": ["Frontier exhausted."],
        }

    # Check depth limit
    if entry.depth > state["max_depth"]:
        await ctx.event_bus.log(
            f"Skipping {entry.url} — exceeds max depth ({entry.depth} > {state['max_depth']})"
        )
        # Return to pick_url by keeping should_continue True
        return {
            "current_url": "",
            "navigation_ok": False,
            "log_messages": [f"Depth limit skip: {entry.url}"],
        }

    ctx.frontier.mark_visited(entry.url)

    await ctx.event_bus.log(
        f"Picked URL (depth={entry.depth}, conf={-entry.priority:.2f}): {entry.url}"
    )

    return {
        "current_url": entry.url,
        "current_depth": entry.depth,
        "navigation_ok": False,
        "error": None,
        "log_messages": [f"→ {entry.url}"],
    }


# ─── Node: navigate ──────────────────────────────────────────────
async def navigate(state: CrawlerState) -> dict:
    """Navigate the browser to the current URL."""
    ctx = _ctx
    url = state["current_url"]

    if not url:
        return {"navigation_ok": False}

    success = await run_with_timeout(
        ctx.browser.navigate(url),
        timeout=CONFIG.browser.navigation_timeout_ms / 1000 + 5,
        fallback=False,
        label=f"navigate({url[:60]})",
    )

    if not success:
        ctx.watchdog.record_stall()
        await ctx.event_bus.error(f"Navigation failed: {url}", url)
        return {
            "navigation_ok": False,
            "error": f"Nav failed: {url}",
            "log_messages": [f"✗ Navigation failed: {url}"],
        }

    await ctx.event_bus.log(f"Loaded: {url}", url)
    return {
        "navigation_ok": True,
        "log_messages": [f"✓ Loaded: {url}"],
    }


# ─── Node: parse ─────────────────────────────────────────────────
async def parse(state: CrawlerState) -> dict:
    """
    1. CHECK FOR CAPTCHA/BOT BLOCK FIRST — skip instantly if detected
    2. Quick quality check — skip pages with no product signals
    3. Light scroll (4 viewports max) to load lazy content
    4. Capture text, screenshot, links
    """
    ctx = _ctx
    url = state["current_url"]

    # ── STEP 0: Captcha/bot detection (instant, no LLM cost) ──
    block_reason = await ctx.browser.detect_block_or_captcha()
    if block_reason:
        await ctx.event_bus.log(f"BLOCKED — {block_reason}. Skipping page.", url)
        return {
            "raw_html": "",
            "viewport_text": "",
            "full_page_text": "",
            "screenshot_bytes": b"",
            "table_contexts": [],
            "page_links": [],
            "page_has_data": False,
            "log_messages": [f"Blocked: {block_reason}"],
        }

    # ── STEP 1: Quick quality signal (no LLM, just JS check) ──
    quality = await ctx.browser.get_page_quality_signal()
    if quality["text_length"] < 200:
        await ctx.event_bus.log(f"Page too short ({quality['text_length']} chars). Skipping.", url)
        return {
            "raw_html": "", "viewport_text": "", "full_page_text": "",
            "screenshot_bytes": b"", "table_contexts": [], "page_links": [],
            "page_has_data": False,
            "log_messages": [f"Too short: {quality['text_length']} chars"],
        }

    # ── STEP 2: Light scroll to trigger lazy content (4 viewports max) ──
    scroll_steps = await ctx.browser.scroll_to_load_all()
    if scroll_steps > 1:
        await ctx.event_bus.log(f"Scrolled {scroll_steps} viewports", url)

    # ── STEP 3: Capture content ──
    viewport_text, table_contexts = await ctx.browser.get_viewport_text_with_table_context()
    screenshot_bytes = await ctx.browser.capture_viewport_screenshot()
    full_page_text = await ctx.browser.get_full_page_text()

    raw_html = await ctx.browser.get_raw_html()
    base_url = await ctx.browser.get_current_url()
    page_links = ctx.parser.extract_links_with_context(
        raw_html, base_url, max_links=CONFIG.crawler.max_links_per_page
    )

    text_len = len(full_page_text)
    links_count = len(page_links)
    has_prices = quality.get("has_prices", False)

    msg = f"Parsed: {text_len} chars, {links_count} links, prices={'YES' if has_prices else 'NO'}"
    await ctx.event_bus.log(msg, url)

    return {
        "raw_html": raw_html,
        "viewport_text": viewport_text,
        "full_page_text": full_page_text,
        "screenshot_bytes": screenshot_bytes,
        "table_contexts": table_contexts,
        "page_links": page_links,
        "log_messages": [msg],
    }


# ─── Node: classify ──────────────────────────────────────────────
async def classify(state: CrawlerState) -> dict:
    """
    Phase 2: Quick triage — 2 viewports max (was 4).
    If parse already marked page_has_data as False (captcha/blocked), skip entirely.
    """
    ctx = _ctx
    url = state["current_url"]

    # Skip if parse already rejected the page (captcha/too short)
    if not state.get("viewport_text", "").strip():
        return {
            "page_has_data": False,
            "log_messages": ["Triage: skipped (no content from parse)"],
        }

    max_triage_viewports = 2  # Reduced from 4 — saves 10+ seconds

    for vp_idx in range(max_triage_viewports):
        if vp_idx > 0:
            scrolled = await ctx.browser.scroll_down_one_viewport()
            if not scrolled:
                break
            await ctx.event_bus.log(f"Triage: checking viewport {vp_idx + 1}...", url)

        viewport_text, _ = await ctx.browser.get_viewport_text_with_table_context()
        screenshot_bytes = await ctx.browser.capture_viewport_screenshot()

        if not viewport_text.strip():
            continue

        has_data = await run_with_timeout(
            ctx.classifier.classify(
                viewport_text=viewport_text,
                target_query=state["target_query"],
                screenshot_bytes=screenshot_bytes,
            ),
            timeout=CONFIG.llm.llm_call_timeout,
            fallback=False,
            label=f"classify_vp{vp_idx}",
        )

        if has_data:
            await ctx.event_bus.log(
                f"Triage: YES — data found on viewport {vp_idx + 1}", url
            )
            return {
                "page_has_data": True,
                "log_messages": [f"Triage: YES (viewport {vp_idx + 1})"],
            }

    await ctx.event_bus.log(f"Triage: NO — no data found", url)
    return {
        "page_has_data": False,
        "log_messages": ["Triage: NO"],
    }


# ─── Node: extract ───────────────────────────────────────────────
async def extract(state: CrawlerState) -> dict:
    """Phase 3: Extract structured data from the page."""
    ctx = _ctx
    mode = state["mode"]
    url = state["current_url"]

    await ctx.event_bus.log(f"Extracting ({mode})...", url)

    if mode == "TEXT_ONLY":
        items = await run_with_timeout(
            ctx.extractor.extract_text_mode(
                full_page_text=state["full_page_text"],
                target_query=state["target_query"],
                schema_class=state["schema_class"],
            ),
            timeout=CONFIG.llm.llm_call_timeout + 10,
            fallback=[],
            label="extract_text",
        )
    elif mode == "SINGLE_IMAGE":
        items = await run_with_timeout(
            ctx.extractor.extract_image_mode(
                page=ctx.browser.page,
                full_page_text=state["full_page_text"],
                target_query=state["target_query"],
                schema_class=state["schema_class"],
            ),
            timeout=CONFIG.llm.llm_call_timeout * 3,  # Image mode is slower
            fallback=[],
            label="extract_image",
        )
    else:
        items = []

    count = len(items) if items else 0
    await ctx.event_bus.log(f"LLM returned {count} raw items", url)

    return {
        "extracted_items": items or [],
        "log_messages": [f"Extracted {count} raw items"],
    }


# ─── Node: validate ──────────────────────────────────────────────
async def validate(state: CrawlerState) -> dict:
    """Filter out junk items (mostly nulls/zeros)."""
    ctx = _ctx
    items = state.get("extracted_items", [])

    if not items:
        return {"validated_items": [], "log_messages": ["No items to validate"]}

    valid = ctx.validator.validate_batch(items)
    rejected = len(items) - len(valid)

    msg = f"Validated: {len(valid)} passed, {rejected} rejected (junk)"
    if rejected > 0:
        await ctx.event_bus.log(msg, state["current_url"])

    return {
        "validated_items": valid,
        "log_messages": [msg],
    }


# ─── Node: dedupe ────────────────────────────────────────────────
async def dedupe(state: CrawlerState) -> dict:
    """Run validated items through SimHash + RapidFuzz deduplication."""
    ctx = _ctx
    items = state.get("validated_items", [])
    url = state["current_url"]

    unique = []
    dup_count = 0

    for item in items:
        data = item.model_dump() if hasattr(item, "model_dump") else item
        if ctx.deduplicator.is_duplicate(data):
            dup_count += 1
        else:
            unique.append(item)

    if dup_count > 0:
        await ctx.event_bus.log(f"Dedup: {dup_count} duplicates discarded", url)

    return {
        "unique_items": unique,
        "log_messages": [f"Unique: {len(unique)}, Duplicates: {dup_count}"],
    }


# ─── Node: emit ──────────────────────────────────────────────────
async def emit(state: CrawlerState) -> dict:
    """Emit unique items to event bus and update counters."""
    ctx = _ctx
    items = state.get("unique_items", [])
    url = state["current_url"]
    total = state.get("total_extracted", 0)

    for item in items:
        data = item.model_dump() if hasattr(item, "model_dump") else item
        await ctx.event_bus.data(data, url)
        total += 1
        await ctx.event_bus.log(f"✓ Item #{total}: {data}", url)

    if items:
        ctx.watchdog.record_extraction(len(items))
        await ctx.event_bus.log(
            f"Page complete: {len(items)} new items (total: {total})", url
        )
    else:
        ctx.watchdog.record_stall()

    ctx.watchdog.record_page_processed()

    return {
        "total_extracted": total,
        "pages_processed": state.get("pages_processed", 0) + 1,
        "stall_count": 0 if items else state.get("stall_count", 0) + 1,
        "log_messages": [f"Emitted {len(items)} items, total={total}"],
    }


# ─── Node: steer_links ───────────────────────────────────────────
async def steer_links(state: CrawlerState) -> dict:
    """
    Phase 1: Score links and push best ones to frontier.
    ALWAYS runs — even after extraction (fixes the pagination bug).
    """
    ctx = _ctx
    links = state.get("page_links", [])
    url = state["current_url"]
    depth = state.get("current_depth", 1)

    if not links:
        return {"scored_links": [], "log_messages": ["No links to score"]}

    # Ask main LLM to score links by relevance
    scored = await run_with_timeout(
        ctx.llm.score_links(links, state["target_query"]),
        timeout=CONFIG.llm.llm_call_timeout,
        fallback=[],
        label="score_links",
    )

    if not scored:
        return {"scored_links": [], "log_messages": ["Link scoring returned nothing"]}

    # Push scored links to priority frontier
    pushed = 0
    for score_entry in scored:
        link_id = score_entry.get("id")
        confidence = score_entry.get("confidence", 0.3)

        # Find the actual URL for this ID
        matching = [l for l in links if l["id"] == link_id]
        if matching and confidence > 0.2:  # Skip very low confidence
            added = ctx.frontier.add(
                url=matching[0]["url"],
                depth=depth + 1,
                confidence=confidence,
                source_url=url,
                base_url=url,
            )
            if added:
                pushed += 1

    msg = f"Steered {pushed} links into frontier (from {len(scored)} scored)"
    await ctx.event_bus.log(msg, url)

    return {
        "scored_links": scored,
        "log_messages": [msg],
    }


# ─── Node: check_continue ────────────────────────────────────────
async def check_continue(state: CrawlerState) -> dict:
    """
    Decide whether to continue crawling or stop.
    Checks: frontier empty? Too many stalls? Watchdog triggered?
    """
    ctx = _ctx

    # Check watchdog
    if ctx.watchdog.should_abandon_branch():
        await ctx.event_bus.log(
            f"Watchdog: {ctx.watchdog.stats['consecutive_stalls']} consecutive stalls — "
            f"abandoning branch."
        )
        ctx.frontier.clear_queue()
        ctx.watchdog.reset_for_new_branch()
        # Will fall through to frontier.pop() which returns None if truly empty

    should_continue = not ctx.frontier.is_empty or ctx.frontier.size > 0

    if not should_continue:
        await ctx.event_bus.log(
            f"Crawl complete. Total: {state.get('total_extracted', 0)} items "
            f"from {state.get('pages_processed', 0)} pages."
        )
        await ctx.event_bus.status("completed")

    return {"should_continue": should_continue}
