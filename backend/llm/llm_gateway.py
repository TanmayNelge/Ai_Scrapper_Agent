"""
Main LLM Gateway — handles extraction and link scoring.
Key improvements over the original llm_client.py:
1. Retry with JSON repair prompt on parse failure
2. Response cache (hash of prompt → cached result)
3. Token-aware truncation (not character-based)
4. Separated from binary decisions (those go to BinaryLLM)
"""
import json
import hashlib
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Any, Type, Optional
from ..config import CONFIG
from ..utils.watchdog import retry_async
from ..llm.prompt_registry import Prompts
from ..utils.schema_builder import schema_fields_description


class LinkScore(BaseModel):
    id: str
    confidence: float


class LinkScoringResult(BaseModel):
    links: list[dict]


class ExtractionList(BaseModel):
    items: list[dict]


class LLMGateway:
    """Main LLM for extraction and link scoring. NOT for binary YES/NO."""

    def __init__(self):
        cfg = CONFIG.llm
        self.client = AsyncOpenAI(
            base_url=cfg.main_base_url,
            api_key=cfg.main_api_key,
        )
        self.model = cfg.main_model
        self.temperature = cfg.main_temperature
        self.max_tokens = cfg.main_max_tokens
        self.context_window = cfg.main_context_window
        self.timeout = cfg.llm_call_timeout

        # Response cache: hash(system+user prompt) → parsed result
        self._cache: dict[str, Any] = {}

    def _cache_key(self, messages: list[dict]) -> str:
        """Create a hash key from the message content for caching."""
        content_str = ""
        for msg in messages:
            c = msg.get("content", "")
            if isinstance(c, str):
                content_str += c
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_str += part.get("text", "")
        return hashlib.md5(content_str.encode()).hexdigest()

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """
        Smart truncation that respects line boundaries.
        Approximate: 1 token ≈ 3.5 chars for English text.
        """
        if len(text) <= max_chars:
            return text

        # Cut at last complete line before the limit
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.7:
            truncated = truncated[:last_newline]

        return truncated + "\n[...truncated...]"

    async def _call_llm(
        self,
        messages: list[dict],
        response_format: dict,
        tag: str,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Core LLM call with:
        - Caching (skip identical requests)
        - Retry with exponential backoff
        - JSON repair on parse failure
        """
        # Check cache
        if use_cache:
            key = self._cache_key(messages)
            if key in self._cache:
                return self._cache[key]

        async def _attempt():
            try:
                # Try json_schema first (better enforcement)
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                if "json_schema" in str(e).lower() or "response_format" in str(e).lower():
                    # Fallback to simpler json_object format
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    raise
            raw = resp.choices[0].message.content
            # Validate it's parseable JSON
            json.loads(raw)
            return raw

        raw = await retry_async(
            _attempt,
            max_retries=CONFIG.llm.max_retries,
            delay=CONFIG.llm.retry_delay,
            timeout=self.timeout,
            label=f"LLMGateway.{tag}",
        )

        if raw is None:
            return None

        # Try JSON repair if initial parse failed but we got text back
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            raw = await self._attempt_json_repair(messages, raw, str(e), response_format, tag)

        # Cache the result
        if use_cache and raw:
            key = self._cache_key(messages)
            self._cache[key] = raw

        return raw

    async def _attempt_json_repair(
        self, original_messages, bad_response, error_msg, response_format, tag
    ) -> Optional[str]:
        """Send the bad JSON back to the LLM with a repair prompt."""
        repair_messages = original_messages + [
            {"role": "assistant", "content": bad_response},
            {
                "role": "user",
                "content": Prompts.JSON_REPAIR.format(error=error_msg),
            },
        ]
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=repair_messages,
                response_format=response_format,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            repaired = resp.choices[0].message.content
            json.loads(repaired)  # Validate
            print(f"[LLMGateway] JSON repair succeeded for {tag}")
            return repaired
        except Exception as e:
            print(f"[LLMGateway] JSON repair failed for {tag}: {e}")
            return None

    # ─── Phase 1: Link Scoring ────────────────────────────────────
    async def score_links(
        self, links_context: list[dict], target_query: str
    ) -> list[dict]:
        """
        Score links by relevance confidence (0.0–1.0).
        Returns: [{"id": "0", "confidence": 0.9}, ...]
        """
        if not links_context:
            return []

        context_str = "\n".join(
            f"ID: {item['id']} | Anchor: {item['anchor_text']} | Context: {item.get('context', '')}"
            for item in links_context
        )

        system = Prompts.LINK_STEERING_SYSTEM.format(target_query=target_query)
        user = Prompts.LINK_STEERING_USER.format(links_context=context_str)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._truncate_text(user, 4000)},
        ]

        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "link_scoring",
                "strict": True,
                "schema": LinkScoringResult.model_json_schema(),
            },
        }

        raw = await self._call_llm(messages, fmt, "score_links", use_cache=False)
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            results = parsed.get("links", [])
            # Validate and clamp confidence
            clean = []
            for item in results:
                if isinstance(item, dict) and "id" in item:
                    conf = float(item.get("confidence", 0.5))
                    conf = max(0.0, min(1.0, conf))
                    clean.append({"id": str(item["id"]), "confidence": conf})
            return clean
        except Exception as e:
            print(f"[LLMGateway] Link scoring parse error: {e}")
            return []

    # ─── Phase 3a: Text Extraction ────────────────────────────────
    async def extract_data(
        self, text: str, target_query: str, schema_class: Type[BaseModel]
    ) -> list[Any]:
        """
        Extract ALL matching items from page text.
        Returns a list of Pydantic model instances.
        """
        fields_desc = schema_fields_description(schema_class)
        max_chars = CONFIG.crawler.max_text_chars_for_llm

        system = Prompts.EXTRACT_TEXT_SYSTEM.format(
            target_query=target_query, fields_desc=fields_desc
        )
        user = Prompts.EXTRACT_TEXT_USER.format(
            text=self._truncate_text(text, max_chars)
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "data_extraction",
                "strict": True,
                "schema": ExtractionList.model_json_schema(),
            },
        }

        raw = await self._call_llm(messages, fmt, "extract_data", use_cache=False)
        if not raw:
            return []

        results = []
        try:
            parsed = json.loads(raw)
            for item_dict in parsed.get("items", []):
                try:
                    results.append(schema_class(**item_dict))
                except Exception as e:
                    print(f"[LLMGateway] Skipping invalid item: {e}")
        except Exception as e:
            print(f"[LLMGateway] Extraction parse error: {e}")

        return results

    # ─── Phase 3b: Single Image Extraction ────────────────────────
    async def extract_image_data(
        self,
        source_img_b64: str,
        context_img_b64: str,
        page_text: str,
        schema_class: Type[BaseModel],
    ) -> Optional[Any]:
        """Extract data from an image with page context."""
        messages = [
            {"role": "system", "content": Prompts.EXTRACT_IMAGE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Page context:\n{page_text[:2000]}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{source_img_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{context_img_b64}"},
                    },
                ],
            },
        ]

        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "image_extraction",
                "strict": True,
                "schema": schema_class.model_json_schema(),
            },
        }

        raw = await self._call_llm(messages, fmt, "extract_image", use_cache=False)
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            return schema_class(**parsed)
        except Exception as e:
            print(f"[LLMGateway] Image extraction parse error: {e}")
            return None

    def clear_cache(self):
        self._cache.clear()
