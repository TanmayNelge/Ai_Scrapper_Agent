# """
# Central configuration for the AI Scraper.
# Tuned for Victus 3050/4050 running LM Studio with Qwen models.
# """
# from dataclasses import dataclass, field


# @dataclass
# class BrowserConfig:
#     headless: bool = False
#     viewport_width: int = 1280
#     viewport_height: int = 800
#     navigation_timeout_ms: int = 25_000
#     post_nav_delay: float = 1.5
#     scroll_overlap: float = 0.8
#     post_scroll_delay: float = 0.5        # Was 1.5 — saves 4-6 sec per page
#     screenshot_quality: int = 60
#     max_viewports_per_page: int = 4       # Was 8 — product data is always near top


# @dataclass
# class LLMConfig:
#     main_base_url: str = "http://127.0.0.1:1234/v1"
#     main_api_key: str = "lm-studio"
#     main_model: str = "local-model"
#     main_temperature: float = 0.1
#     main_max_tokens: int = 1024
#     main_context_window: int = 8192

#     binary_base_url: str = "http://127.0.0.1:1235/v1"
#     binary_api_key: str = "lm-studio"
#     binary_model: str = "local-binary-model"
#     binary_temperature: float = 0.0
#     binary_max_tokens: int = 32

#     vision_enabled: bool = True

#     max_retries: int = 2
#     retry_delay: float = 0.5
#     llm_call_timeout: float = 90.0


# @dataclass
# class CrawlerConfig:
#     max_depth: int = 3
#     max_seeds: int = 50
#     max_links_per_page: int = 15
#     stall_timeout_seconds: float = 45.0
#     max_consecutive_stalls: int = 5
#     max_text_chars_for_llm: int = 2000
#     max_text_chars_for_triage: int = 1200


# @dataclass
# class DeduplicatorConfig:
#     simhash_threshold: int = 5            # Wider candidate net
#     fuzz_threshold: float = 75.0          # Tighter final check — catches more dupes


# @dataclass
# class AppConfig:
#     browser: BrowserConfig = field(default_factory=BrowserConfig)
#     llm: LLMConfig = field(default_factory=LLMConfig)
#     crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
#     dedup: DeduplicatorConfig = field(default_factory=DeduplicatorConfig)
#     db_url: str = "sqlite:///./scraper_data.db"
#     log_level: str = "INFO"


# CONFIG = AppConfig()
"""
Central configuration for the AI Scraper.
All timeouts, model endpoints, thresholds, and operational limits.
Tuned for Victus 3050/4050 running LM Studio with Qwen models.
"""
from dataclasses import dataclass, field


@dataclass
class BrowserConfig:
    headless: bool = False                # HEADFUL mode for reliability
    viewport_width: int = 1280
    viewport_height: int = 800
    navigation_timeout_ms: int = 30_000
    post_nav_delay: float = 2.0           # Human-like delay after navigation
    scroll_overlap: float = 0.8           # Scroll 80% of viewport
    post_scroll_delay: float = 1.5
    screenshot_quality: int = 60          # JPEG quality
    max_viewports_per_page: int = 15      # Safety cap


@dataclass
class LLMConfig:
    # Main LLM (Qwen 3.5 4B) — extraction + link scoring
    main_base_url: str = "http://127.0.0.1:1234/v1"
    main_api_key: str = "lm-studio"
    main_model: str = "qwen/qwen3-vl-4b"
    main_temperature: float = 0.1
    main_max_tokens: int = 4096
    main_context_window: int = 8192       # Safe limit for 4B on 3050

    # Binary LLM (small model) — YES/NO triage only
    binary_base_url: str = "http://127.0.0.1:1234/v1"
    binary_api_key: str = "lm-studio"
    binary_model: str = "smolvlm-500m-instruct"
    binary_temperature: float = 0.0       # Deterministic for binary
    binary_max_tokens: int = 32           # Only needs YES or NO

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    llm_call_timeout: float = 60.0        # Kill LLM call after 60s


@dataclass
class CrawlerConfig:
    max_depth: int = 3
    max_seeds: int = 100
    max_links_per_page: int = 30          # Cap links sent to LLM
    stall_timeout_seconds: float = 45.0   # No progress = stall
    max_consecutive_stalls: int = 3       # 3 stalls = abandon branch
    max_text_chars_for_llm: int = 5000    # Char limit for extraction prompt
    max_text_chars_for_triage: int = 2500


@dataclass
class DeduplicatorConfig:
    simhash_threshold: int = 3            # Max hamming distance
    fuzz_threshold: float = 85.0          # RapidFuzz ratio %


@dataclass
class AppConfig:
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    dedup: DeduplicatorConfig = field(default_factory=DeduplicatorConfig)
    db_url: str = "sqlite:///./scraper_data.db"
    log_level: str = "INFO"


# Global singleton
CONFIG = AppConfig()