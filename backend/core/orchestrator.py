"""
Orchestrator — the top-level controller that:
1. Creates all components (browser, LLMs, frontier, etc.)
2. Fetches seeds from DuckDuckGo
3. Runs the LangGraph state machine
4. Handles graceful shutdown

This replaces the old monolithic engine.py.
"""
import asyncio
from typing import Optional

from ..config import CONFIG
from ..browser.browser_manager import BrowserManager
from ..browser.dom_parser import DOMParser
from ..llm.llm_gateway import LLMGateway
from ..llm.binary_llm import BinaryLLM
from ..pipeline.classifier import PageClassifier
from ..pipeline.extractor import DataExtractor
from ..pipeline.validator import DataValidator
from ..pipeline.deduplicator import Deduplicator
from ..utils.url_frontier import URLFrontier
from ..utils.watchdog import Watchdog
from ..utils.event_bus import EventBus
from ..utils.schema_builder import build_dynamic_model
from ..seeds.duckduckgo import get_seeds_async
from .nodes import NodeContext, set_node_context
from .graph import compile_crawl_graph
from .state import CrawlerState


class Orchestrator:
    """
    High-level controller for a single scraping project.
    One Orchestrator per project. Manages lifecycle: init → run → stop.
    """

    def __init__(
        self,
        project_id: int,
        mode: str,
        query: str,
        json_schema_dict: dict,
        max_depth: int = None,
    ):
        self.project_id = project_id
        self.mode = mode.upper()  # TEXT_ONLY or SINGLE_IMAGE
        self.query = query
        self.max_depth = max_depth or CONFIG.crawler.max_depth

        # Build runtime Pydantic schema
        self.schema_class = build_dynamic_model(
            f"Project{project_id}Schema", json_schema_dict
        )

        # Core components
        self.browser = BrowserManager()
        self.dom_parser = DOMParser()
        self.llm_gateway = LLMGateway()
        self.binary_llm = BinaryLLM()
        self.classifier = PageClassifier(self.binary_llm)
        self.extractor = DataExtractor(self.llm_gateway, self.binary_llm)
        self.validator = DataValidator(min_filled_ratio=0.3)
        self.deduplicator = Deduplicator()
        self.frontier = URLFrontier()
        self.watchdog = Watchdog(
            stall_timeout=CONFIG.crawler.stall_timeout_seconds,
            max_consecutive_stalls=CONFIG.crawler.max_consecutive_stalls,
        )
        self.event_bus = EventBus(project_id)

        # LangGraph compiled graph
        self._graph = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Initialize browser, fetch seeds, start the graph."""
        self._running = True
        await self.event_bus.status("starting")
        await self.event_bus.log(f"Starting project '{self.query}' in {self.mode} mode")

        # ─── HEALTH CHECK: Verify LLM is loaded before doing anything ───
        await self.event_bus.log("Checking LLM connection...")
        llm_ok = await self._check_llm_health()
        if not llm_ok:
            await self.event_bus.error(
                "LLM HEALTH CHECK FAILED — No model loaded in LM Studio! "
                "Open LM Studio → load a model → click 'Start Server' on port 1234. "
                "Then try again."
            )
            await self.event_bus.status("failed")
            return
        await self.event_bus.log("LLM health check passed — model is responding")

        # Start browser (HEADFUL)
        await self.browser.start()
        await self.event_bus.log("Browser launched (headful mode)")

        # Fetch seeds
        await self.event_bus.log("Fetching seed URLs from DuckDuckGo...")
        seeds = await get_seeds_async(self.query, CONFIG.crawler.max_seeds)
        added = self.frontier.add_seeds(seeds)
        await self.event_bus.log(f"Loaded {added} seed URLs into frontier")

        if added == 0:
            await self.event_bus.error("No seeds found. Cannot start crawling.")
            await self.event_bus.status("failed")
            await self.browser.stop()
            return

        # Wire up the NodeContext so nodes can access shared resources
        ctx = NodeContext(
            browser=self.browser,
            frontier=self.frontier,
            watchdog=self.watchdog,
            event_bus=self.event_bus,
            classifier=self.classifier,
            extractor=self.extractor,
            validator=self.validator,
            deduplicator=self.deduplicator,
            llm_gateway=self.llm_gateway,
            dom_parser=self.dom_parser,
        )
        set_node_context(ctx)

        # Compile graph
        self._graph = compile_crawl_graph()

        # Build initial state
        initial_state: CrawlerState = {
            "project_id": self.project_id,
            "mode": self.mode,
            "target_query": self.query,
            "max_depth": self.max_depth,
            "schema_class": self.schema_class,
            "current_url": "",
            "current_depth": 0,
            "navigation_ok": False,
            "raw_html": "",
            "viewport_text": "",
            "full_page_text": "",
            "screenshot_bytes": b"",
            "table_contexts": [],
            "page_links": [],
            "page_has_data": False,
            "extracted_items": [],
            "validated_items": [],
            "unique_items": [],
            "scored_links": [],
            "should_continue": True,
            "stall_count": 0,
            "total_extracted": 0,
            "pages_processed": 0,
            "error": None,
            "log_messages": [],
        }

        await self.event_bus.status("running")
        await self.event_bus.log("LangGraph state machine started")

        # Run the graph
        try:
            final_state = await self._graph.ainvoke(initial_state)
            total = final_state.get("total_extracted", 0)
            pages = final_state.get("pages_processed", 0)
            await self.event_bus.log(
                f"Crawl finished: {total} items from {pages} pages"
            )
            await self.event_bus.status("completed")
        except asyncio.CancelledError:
            await self.event_bus.log("Crawl cancelled by user")
            await self.event_bus.status("cancelled")
        except Exception as e:
            await self.event_bus.error(f"Crawl crashed: {e}")
            await self.event_bus.status("failed")
        finally:
            self._running = False
            await self.browser.stop()
            await self.event_bus.log("Browser closed. Orchestrator shut down.")

    async def _check_llm_health(self) -> bool:
        """
        Send a tiny test request to LM Studio to verify a model is loaded.
        Returns True if the model responds, False otherwise.
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=CONFIG.llm.main_base_url,
                api_key=CONFIG.llm.main_api_key,
            )
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=CONFIG.llm.main_model,
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=5,
                    temperature=0,
                ),
                timeout=30.0,
            )
            text = resp.choices[0].message.content
            await self.event_bus.log(f"LLM responded: '{text.strip()[:50]}'")
            return True
        except asyncio.TimeoutError:
            await self.event_bus.error("LLM health check timed out after 30s")
            return False
        except Exception as e:
            error_msg = str(e)
            if "No models loaded" in error_msg:
                await self.event_bus.error(
                    "LM Studio is running but NO MODEL IS LOADED. "
                    "Go to LM Studio → search 'qwen2.5-3b-instruct' → download Q4_K_M → "
                    "load it → start the server on port 1234."
                )
            elif "Connection refused" in error_msg or "connect" in error_msg.lower():
                await self.event_bus.error(
                    "Cannot connect to LM Studio on port 1234. "
                    "Make sure LM Studio is running and the local server is started."
                )
            else:
                await self.event_bus.error(f"LLM health check error: {error_msg[:200]}")
            return False

    async def stop(self):
        """Gracefully stop the crawl."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        await self.browser.stop()
        await self.event_bus.status("stopped")
        await self.event_bus.log("Orchestrator stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            **self.watchdog.stats,
            "frontier_size": self.frontier.size,
            "visited_count": self.frontier.visited_count,
            "dedup_store_size": self.deduplicator.count,
            "running": self._running,
        }
