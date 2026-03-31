"""
Browser Manager — Headful Playwright with critical fixes:
1. HEADFUL mode (not headless) for better anti-bot bypass
2. Viewport-aware text extraction (only visible text, not full page)
3. TABLE CONTEXT FIX: When scrolling through a data table, detects
   if column headers have scrolled out of view and injects them
   as context so the LLM always knows what columns it's reading.
"""
import asyncio
from playwright.async_api import async_playwright, Page, BrowserContext
from typing import Optional
from ..config import CONFIG


class BrowserManager:
    def __init__(self):
        self._cfg = CONFIG.browser
        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self._cfg.headless,  # HEADFUL by default
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--disable-default-apps",
            ],
        )
        self.context = await self.browser.new_context(
            viewport={
                "width": self._cfg.viewport_width,
                "height": self._cfg.viewport_height,
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        # Stealth: remove navigator.webdriver flag
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        self.page = await self.context.new_page()

    async def stop(self):
        for resource in (self.context, self.browser, self.playwright):
            if resource:
                try:
                    await resource.close() if hasattr(resource, "close") else await resource.stop()
                except Exception:
                    pass

    async def navigate(self, url: str) -> bool:
        if not self.page:
            return False
        try:
            await self.page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self._cfg.navigation_timeout_ms,
            )
            await asyncio.sleep(self._cfg.post_nav_delay)
            return True
        except Exception as e:
            print(f"[Browser] Navigation failed for {url}: {e}")
            return False

    async def get_current_url(self) -> str:
        return self.page.url if self.page else ""

    async def get_raw_html(self) -> str:
        if not self.page:
            return ""
        try:
            return await self.page.content()
        except Exception:
            return ""

    # ─── THE TABLE CONTEXT FIX ────────────────────────────────────
    async def get_viewport_text_with_table_context(self) -> tuple[str, list[dict]]:
        """
        Extract ONLY visible text from the current viewport,
        plus inject any table column headers that have scrolled out of view.

        Returns:
            (enriched_text, table_contexts)
            where enriched_text has sticky headers prepended if needed.
        """
        if not self.page:
            return "", []

        try:
            result = await self.page.evaluate(r"""() => {
                const vpTop = window.scrollY;
                const vpBottom = vpTop + window.innerHeight;

                // ─── Part 1: Find table headers scrolled above viewport ───
                const tableContexts = [];
                const tables = document.querySelectorAll('table');
                for (const table of tables) {
                    const tRect = table.getBoundingClientRect();
                    const absTop = tRect.top + window.scrollY;
                    const absBottom = tRect.bottom + window.scrollY;

                    // Table body visible but potentially header is above viewport
                    if (absBottom > vpTop && absTop < vpBottom) {
                        const thead = table.querySelector('thead')
                                   || table.querySelector('tr:first-child');
                        if (thead) {
                            const hRect = thead.getBoundingClientRect();
                            const hAbsTop = hRect.top + window.scrollY;
                            // Header is scrolled above viewport
                            if (hAbsTop < vpTop) {
                                const cols = [];
                                thead.querySelectorAll('th, td').forEach(cell => {
                                    const t = cell.innerText.trim();
                                    if (t) cols.push(t);
                                });
                                if (cols.length > 0) {
                                    tableContexts.push({ columns: cols });
                                }
                            }
                        }
                    }
                }

                // ─── Part 2: Extract only text visible in viewport ───
                const texts = [];
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode(node) {
                            const parent = node.parentElement;
                            if (!parent) return NodeFilter.FILTER_REJECT;

                            // Skip hidden elements
                            const style = window.getComputedStyle(parent);
                            if (style.display === 'none'
                                || style.visibility === 'hidden'
                                || parseFloat(style.opacity) === 0) {
                                return NodeFilter.FILTER_REJECT;
                            }

                            // Skip script/style/noscript
                            const tag = parent.tagName.toLowerCase();
                            if (['script','style','noscript','svg'].includes(tag)) {
                                return NodeFilter.FILTER_REJECT;
                            }

                            // Check if element is in viewport
                            const rect = parent.getBoundingClientRect();
                            if (rect.bottom < 0 || rect.top > window.innerHeight) {
                                return NodeFilter.FILTER_REJECT;
                            }

                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );

                let n;
                while (n = walker.nextNode()) {
                    const t = n.textContent.trim();
                    if (t && t.length > 0) texts.push(t);
                }

                return {
                    viewportText: texts.join('\n'),
                    tableContexts: tableContexts
                };
            }""")

            viewport_text = result.get("viewportText", "")
            table_contexts = result.get("tableContexts", [])

            # Inject sticky headers if any table header is out of view
            if table_contexts:
                header_block = "=== TABLE COLUMN HEADERS (scrolled above, still apply) ===\n"
                for ctx in table_contexts:
                    header_block += "Columns: " + " | ".join(ctx["columns"]) + "\n"
                header_block += "=== DATA IN CURRENT VIEWPORT ===\n\n"
                viewport_text = header_block + viewport_text

            return viewport_text, table_contexts

        except Exception as e:
            print(f"[Browser] Viewport text extraction failed: {e}")
            # Fallback: full page text (worse but functional)
            try:
                text = await self.page.inner_text("body")
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return "\n".join(lines[:200]), []  # Cap at 200 lines
            except Exception:
                return "", []

    async def scroll_to_load_all(self) -> int:
        """
        Scroll the ENTIRE page top-to-bottom to trigger lazy loading.
        Returns the number of scroll steps taken.
        Call this BEFORE extracting text so React/Vue content is rendered.
        """
        if not self.page:
            return 0
        try:
            # First scroll to top
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)

            steps = 0
            max_steps = self._cfg.max_viewports_per_page

            for _ in range(max_steps):
                at_bottom = await self.page.evaluate(f"""() => {{
                    const maxScroll = Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight
                    ) - window.innerHeight;
                    const current = window.scrollY;
                    if (current >= maxScroll - 10) return true;
                    window.scrollBy(0, window.innerHeight * {self._cfg.scroll_overlap});
                    return false;
                }}""")
                steps += 1
                # Wait for lazy content to load after each scroll
                await asyncio.sleep(0.8)
                if at_bottom:
                    break

            # Scroll back to top for triage screenshot
            await self.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
            return steps
        except Exception as e:
            print(f"[Browser] Scroll-to-load failed: {e}")
            return 0

    async def detect_block_or_captcha(self) -> str:
        """
        Detect if the page is a captcha, bot block, or access denied page.
        Returns empty string if page is clean, or a reason string if blocked.
        Checks BEFORE wasting LLM calls on junk pages.
        """
        if not self.page:
            return "no page"
        try:
            result = await self.page.evaluate(r"""() => {
                const text = document.body ? document.body.innerText.toLowerCase() : '';
                const title = document.title.toLowerCase();
                const url = window.location.href.toLowerCase();
                const len = text.length;

                // Page too short = likely blocked or redirect stub
                if (len < 100) return 'page_too_short';

                // Captcha indicators
                const captchaWords = [
                    'captcha', 'recaptcha', 'hcaptcha', 'cloudflare',
                    'verify you are human', 'verify you\'re human',
                    'are you a robot', 'are you human', 'bot detection',
                    'please verify', 'security check', 'checking your browser',
                    'just a moment', 'please wait while we verify',
                    'ray id', 'cf-browser-verification',
                    'enable javascript and cookies', 'unusual traffic',
                ];
                for (const w of captchaWords) {
                    if (text.includes(w) || title.includes(w)) return 'captcha: ' + w;
                }

                // Access denied
                const blockWords = [
                    'access denied', 'forbidden', '403 forbidden',
                    'you have been blocked', 'blocked by', 'not authorized',
                    'page not found', '404 not found', 'error 404',
                    'sorry, you have been blocked',
                    'automated access', 'bot detected',
                ];
                for (const w of blockWords) {
                    if (text.includes(w) || title.includes(w)) return 'blocked: ' + w;
                }

                // Check for captcha iframes
                const iframes = document.querySelectorAll('iframe');
                for (const f of iframes) {
                    const src = (f.src || '').toLowerCase();
                    if (src.includes('captcha') || src.includes('challenge')
                        || src.includes('recaptcha') || src.includes('hcaptcha')) {
                        return 'captcha_iframe';
                    }
                }

                return '';
            }""")
            return result or ""
        except Exception:
            return ""

    async def get_page_quality_signal(self) -> dict:
        """
        Quick check: is this page worth spending LLM calls on?
        Returns text length, has_prices, has_product_signals.
        """
        if not self.page:
            return {"text_length": 0, "has_prices": False, "has_product_signals": False}
        try:
            return await self.page.evaluate(r"""() => {
                const text = document.body ? document.body.innerText : '';
                const lower = text.toLowerCase();
                return {
                    text_length: text.length,
                    has_prices: /\$\d|€\d|₹\d|£\d|\d+\.\d{2}/.test(text),
                    has_product_signals: (
                        lower.includes('add to cart') ||
                        lower.includes('buy now') ||
                        lower.includes('price') ||
                        lower.includes('in stock') ||
                        lower.includes('out of stock') ||
                        lower.includes('rating') ||
                        lower.includes('review') ||
                        lower.includes('specifications') ||
                        lower.includes('product')
                    ),
                };
            }""")
        except Exception:
            return {"text_length": 0, "has_prices": False, "has_product_signals": False}

    async def scroll_to_position(self, viewport_index: int) -> bool:
        """Scroll to a specific viewport position (0-indexed)."""
        if not self.page:
            return False
        try:
            offset = int(viewport_index * self._cfg.viewport_height * self._cfg.scroll_overlap)
            await self.page.evaluate(f"window.scrollTo(0, {offset})")
            await asyncio.sleep(0.6)
            return True
        except Exception:
            return False

    async def get_full_page_text(self) -> str:
        """Get ALL text from the page (for extraction after triage passes)."""
        if not self.page:
            return ""
        try:
            text = await self.page.inner_text("body")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)
        except Exception:
            return ""

    async def capture_viewport_screenshot(self) -> bytes:
        if not self.page:
            return b""
        try:
            return await self.page.screenshot(
                type="jpeg", quality=self._cfg.screenshot_quality
            )
        except Exception as e:
            print(f"[Browser] Screenshot failed: {e}")
            return b""

    async def scroll_down_one_viewport(self) -> bool:
        """Scroll down ~80% of viewport. Returns False if already at bottom."""
        if not self.page:
            return False
        try:
            at_bottom = await self.page.evaluate(f"""() => {{
                const maxScroll = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight
                ) - window.innerHeight;
                const current = window.scrollY;
                if (current >= maxScroll - 10) return true;
                window.scrollBy(0, window.innerHeight * {self._cfg.scroll_overlap});
                return false;
            }}""")
            await asyncio.sleep(self._cfg.post_scroll_delay)
            return not at_bottom
        except Exception:
            return False

    async def get_scroll_position(self) -> dict:
        """Get current scroll position for progress tracking."""
        if not self.page:
            return {"current": 0, "max": 0, "percent": 0}
        try:
            return await self.page.evaluate("""() => {
                const max = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight
                ) - window.innerHeight;
                const current = window.scrollY;
                return {
                    current: Math.round(current),
                    max: Math.round(max),
                    percent: max > 0 ? Math.round((current / max) * 100) : 100
                };
            }""")
        except Exception:
            return {"current": 0, "max": 0, "percent": 0}
