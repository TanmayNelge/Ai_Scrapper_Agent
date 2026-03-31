"""
Phase 2: Page Classifier — decides if a page contains target data.
Uses the Binary LLM (separate lightweight model) for fast YES/NO.
"""
from ..llm.binary_llm import BinaryLLM
from ..llm.prompt_registry import Prompts
from ..utils.image_processor import ImageProcessor
from typing import Optional


class PageClassifier:
    def __init__(self, binary_llm: BinaryLLM):
        self.llm = binary_llm
        self.img_proc = ImageProcessor()

    async def classify(
        self,
        viewport_text: str,
        target_query: str,
        screenshot_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        Returns True if the current viewport likely contains target data.
        Only processes screenshots if vision is enabled in config.
        """
        from ..config import CONFIG

        screenshot_b64 = None
        if screenshot_bytes and CONFIG.llm.vision_enabled:
            screenshot_b64 = self.img_proc.screenshot_to_b64(screenshot_bytes)

        if screenshot_b64:
            system = Prompts.TRIAGE_SYSTEM.format(target_query=target_query)
        else:
            system = Prompts.TRIAGE_TEXT_ONLY_SYSTEM.format(target_query=target_query)

        return await self.llm.triage_page(
            text=viewport_text,
            target_query=target_query,
            screenshot_b64=screenshot_b64,
            system_prompt=system,
        )
