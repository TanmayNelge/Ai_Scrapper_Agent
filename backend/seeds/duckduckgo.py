"""
DuckDuckGo Lite seed generator with smart query enhancement and domain filtering.
Adds shopping intent to queries and filters out known non-product domains.
"""
import asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Domains that NEVER have product/price data
BLOCKED_DOMAINS = {
    "wikipedia.org", "reddit.com", "quora.com", "youtube.com",
    "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
    "linkedin.com", "pinterest.com", "medium.com",
    "verywellhealth.com", "healthline.com", "webmd.com", "mayoclinic.org",
    "nih.gov", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "harvard.edu", "stanford.edu", "mit.edu",
    "bbc.com", "cnn.com", "nytimes.com", "wsj.com",
    "stackoverflow.com", "github.com", "arxiv.org",
}

# Domains that are HIGH VALUE for product data
PRIORITY_DOMAINS = {
    "amazon.com", "walmart.com", "ebay.com", "target.com",
    "bestbuy.com", "newegg.com", "costco.com",
    "flipkart.com", "aliexpress.com", "alibaba.com",
    "etsy.com", "shopify.com", "homedepot.com", "lowes.com",
    "bhphotovideo.com", "adorama.com", "microcenter.com",
}


def _is_domain_blocked(url: str) -> bool:
    """Check if URL belongs to a blocked domain."""
    try:
        host = urlparse(url).netloc.lower()
        for blocked in BLOCKED_DOMAINS:
            if host == blocked or host.endswith("." + blocked):
                return True
    except Exception:
        pass
    return False


def _domain_priority(url: str) -> float:
    """Return a priority score — higher = more likely to have product data."""
    try:
        host = urlparse(url).netloc.lower()
        for priority in PRIORITY_DOMAINS:
            if host == priority or host.endswith("." + priority):
                return 0.9
        # URLs with shopping signals in path
        path = urlparse(url).path.lower()
        if any(w in path for w in ["/product", "/buy", "/shop", "/price", "/dp/", "/ip/"]):
            return 0.8
    except Exception:
        pass
    return 0.5


def _enhance_query(query: str) -> list[str]:
    """
    Generate multiple search queries to get better product results.
    Returns [shopping_query, original_query] so shopping results come first.
    """
    lower = query.lower()
    # If user already included shopping intent, don't add more
    has_intent = any(w in lower for w in ["price", "buy", "shop", "cost", "cheap", "deal", "store"])

    queries = []
    if not has_intent:
        queries.append(f"{query} price buy online")
    queries.append(query)
    return queries


def _fetch_seeds_sync(query: str, max_results: int = 50) -> list[str]:
    """Synchronous DDG scraper with smart query enhancement."""
    urls = []
    base_url = "https://lite.duckduckgo.com/lite/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://lite.duckduckgo.com/",
    }

    search_queries = _enhance_query(query)

    for sq in search_queries:
        if len(urls) >= max_results:
            break

        payload = {"q": sq}
        page = 0

        try:
            while len(urls) < max_results and page < 3:
                resp = requests.post(base_url, headers=headers, data=payload, timeout=15)
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                result_links = soup.find_all("a", class_="result-link")

                if not result_links:
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        if href.startswith("http") and "duckduckgo.com" not in href:
                            if not _is_domain_blocked(href) and href not in urls:
                                urls.append(href)
                                if len(urls) >= max_results:
                                    break
                    break

                for link in result_links:
                    href = link.get("href")
                    if href and href.startswith("http") and "duckduckgo.com" not in href:
                        if not _is_domain_blocked(href) and href not in urls:
                            urls.append(href)
                        if len(urls) >= max_results:
                            break

                # Pagination
                next_payload = {}
                for form in soup.find_all("form", action="/lite/"):
                    submit = form.find("input", {"type": "submit", "value": "Next"})
                    if submit:
                        for inp in form.find_all("input"):
                            name = inp.get("name")
                            if name:
                                next_payload[name] = inp.get("value", "")
                        break

                if next_payload:
                    payload = next_payload
                    page += 1
                else:
                    break

        except Exception as e:
            print(f"[Seeds] Error for query '{sq}': {e}")

    # Sort by domain priority — shopping sites first
    urls.sort(key=lambda u: -_domain_priority(u))

    blocked_count = 0
    # One final filter pass
    clean = []
    for u in urls:
        if _is_domain_blocked(u):
            blocked_count += 1
        else:
            clean.append(u)

    if blocked_count:
        print(f"[Seeds] Filtered out {blocked_count} irrelevant domains")
    print(f"[Seeds] DDG returned {len(clean)} product-focused URLs for '{query}'")
    return clean


async def get_seeds_async(query: str, max_results: int = 50) -> list[str]:
    return await asyncio.to_thread(_fetch_seeds_sync, query, max_results)
