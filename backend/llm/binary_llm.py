"""
Binary LLM client — uses a SEPARATE smaller/faster model for YES/NO decisions.
This keeps the main GPU model free for extraction tasks.

On a 3050/4050 you can run:
- Port 1234: Qwen 3.5 4B (main extraction model)
- Port 1235: Qwen 0.5B or Phi-3-mini (fast binary classifier)

Or use the SAME model on SAME port if you only have one model loaded.
Set binary_base_url = main_base_url in config.py to share.
"""
import json
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional
from ..config import CONFIG
from ..utils.watchdog import retry_async


class BinaryResult(BaseModel):
    has_data: bool


class ImageRelevanceResult(BaseModel):
    is_relevant: bool


class BinaryLLM:
    """Lightweight LLM client for binary YES/NO decisions only."""

    def __init__(self):
        cfg = CONFIG.llm
        self.client = AsyncOpenAI(
            base_url=cfg.binary_base_url,
            api_key=cfg.binary_api_key,
        )
        self.model = cfg.binary_model
        self.temperature = cfg.binary_temperature
        self.max_tokens = cfg.binary_max_tokens
        self.timeout = cfg.llm_call_timeout

    async def _call(self, messages: list[dict], schema_class, tag: str) -> Optional[dict]:
        """Core binary LLM call with JSON output."""

        async def _attempt():
            try:
                # Try json_schema first (better enforcement)
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": tag,
                            "strict": True,
                            "schema": schema_class.model_json_schema(),
                        },
                    },
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception:
                # Fallback to json_object (wider compatibility)
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            raw = resp.choices[0].message.content
            return json.loads(raw)

        result = await retry_async(
            _attempt,
            max_retries=2,  # Binary calls are fast, 2 retries is enough
            delay=0.5,
            timeout=self.timeout,
            label=f"BinaryLLM.{tag}",
        )
        return result

    async def triage_page(
        self,
        text: str,
        target_query: str,
        screenshot_b64: Optional[str] = None,
        system_prompt: str = "",
    ) -> bool:
        """
        Phase 2: Does this page contain target data? Returns True/False.
        Only uses vision if CONFIG.llm.vision_enabled is True.
        """
        from ..config import CONFIG

        # NEVER send screenshots to text-only models — wastes retries
        use_vision = screenshot_b64 and CONFIG.llm.vision_enabled

        if use_vision:
            user_content = [
                {"type": "text", "text": f"Page text (viewport):\n{text[:2000]}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"},
                },
            ]
        else:
            user_content = f"Page text (viewport):\n{text[:2500]}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        result = await self._call(messages, BinaryResult, "page_triage")

        if result is None and use_vision:
            # Multimodal failed — retry text-only
            return await self.triage_page(text, target_query, screenshot_b64=None,
                                           system_prompt=system_prompt)

        return result.get("has_data", False) if result else False

    async def is_image_relevant(
        self,
        context_screenshot_b64: str,
        source_image_b64: str,
        system_prompt: str = "",
    ) -> bool:
        """Phase 3b: Is this image a primary data image (not a logo/icon)?"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{context_screenshot_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{source_image_b64}"},
                    },
                ],
            },
        ]

        result = await self._call(messages, ImageRelevanceResult, "image_relevance")
        return result.get("is_relevant", False) if result else False
