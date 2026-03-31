"""
DOM Parser for link extraction and HTML-to-Markdown conversion.
Key fix: extract_links does NOT use Readability (which strips product grids).
Instead, it scans the full DOM but filters out nav/footer/sidebar heuristically.
"""
from bs4 import BeautifulSoup, Tag
from readability import Document
from markdownify import markdownify as md
from urllib.parse import urljoin
from typing import Optional


# Elements that almost never contain useful product/data links
NOISE_SELECTORS = [
    "nav", "footer", "header",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    ".cookie-banner", ".popup", ".modal", ".overlay",
    "#cookie-consent", ".social-share", ".breadcrumb",
]

# Link patterns to always skip
SKIP_HREF_PATTERNS = [
    "javascript:", "mailto:", "tel:", "data:", "#",
    "login", "signin", "signup", "register", "account",
    "cart", "checkout", "wishlist",
    "facebook.com", "twitter.com", "instagram.com", "youtube.com",
    "linkedin.com", "pinterest.com", "tiktok.com",
]


class DOMParser:

    def clean_html_to_markdown(self, raw_html: str, strip_images: bool = True) -> str:
        """
        Uses Mozilla Readability to isolate main content, then converts to Markdown.
        Falls back to raw body text if Readability fails.
        """
        if not raw_html:
            return ""
        try:
            doc = Document(raw_html)
            clean_html = doc.summary()

            strip_tags = ["img", "a"] if strip_images else ["a"]
            markdown = md(clean_html, strip=strip_tags, heading_style="ATX")
            return markdown.strip()
        except Exception as e:
            print(f"[DOMParser] Readability fallback: {e}")
            try:
                soup = BeautifulSoup(raw_html, "html.parser")
                # Remove known noise
                for sel in NOISE_SELECTORS:
                    for el in soup.select(sel):
                        el.decompose()
                return soup.get_text(separator="\n", strip=True)[:8000]
            except Exception:
                return ""

    def extract_links_with_context(
        self, raw_html: str, base_url: str, max_links: int = 30
    ) -> list[dict]:
        """
        Extract links from the FULL DOM (not Readability-filtered)
        with surrounding text context. Skips noise zones.

        Each link gets:
        - id: sequential ID for LLM reference
        - url: absolute URL
        - anchor_text: the link text
        - context: text from parent/sibling elements (product titles, etc.)
        """
        if not raw_html:
            return []

        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove noise zones
        for selector in NOISE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        # Also remove script/style
        for tag_name in ("script", "style", "noscript", "svg"):
            for el in soup.find_all(tag_name):
                el.decompose()

        links = []
        seen_urls = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "").strip()

            # Skip junk links
            if any(p in href.lower() for p in SKIP_HREF_PATTERNS):
                continue

            # Resolve to absolute
            absolute_url = urljoin(base_url, href)

            # Deduplicate
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)

            anchor_text = a_tag.get_text(strip=True)

            # Context: walk up to parent and grab surrounding text
            context = self._extract_context(a_tag, anchor_text)

            # Only include links that have SOME text signal
            if anchor_text or context:
                links.append({
                    "id": str(len(links)),
                    "url": absolute_url,
                    "anchor_text": anchor_text[:120],
                    "context": context[:150],
                })

            if len(links) >= max_links:
                break

        return links

    def _extract_context(self, a_tag: Tag, anchor_text: str) -> str:
        """
        Walk up the DOM from a link to find descriptive context.
        Looks for headings, strong tags, or parent text.
        """
        parent = a_tag.parent
        if not parent:
            return ""

        # Look for headings in ancestors (up to 3 levels)
        for _ in range(3):
            if parent is None:
                break
            headers = parent.find_all(["h1", "h2", "h3", "h4", "strong", "b"], limit=3)
            if headers:
                header_texts = [h.get_text(strip=True) for h in headers]
                # Remove the anchor text itself to avoid duplication
                header_texts = [t for t in header_texts if t != anchor_text]
                if header_texts:
                    return " | ".join(header_texts)
            parent = parent.parent

        # Fallback: parent's direct text
        parent = a_tag.parent
        if parent:
            parent_text = parent.get_text(strip=True)
            # Remove anchor text from parent text
            cleaned = parent_text.replace(anchor_text, "").strip()
            return cleaned[:150] if cleaned else ""

        return ""

    def extract_table_structure(self, raw_html: str) -> list[dict]:
        """
        Detect data tables and extract their header structure.
        Used by the pipeline to understand page layout.
        """
        soup = BeautifulSoup(raw_html, "html.parser")
        tables = []
        for table in soup.find_all("table"):
            headers = []
            thead = table.find("thead") or table.find("tr")
            if thead:
                for cell in thead.find_all(["th", "td"]):
                    text = cell.get_text(strip=True)
                    if text:
                        headers.append(text)

            row_count = len(table.find_all("tr"))
            if headers and row_count > 1:
                tables.append({
                    "headers": headers,
                    "row_count": row_count,
                    "has_data": row_count > 2,
                })
        return tables
