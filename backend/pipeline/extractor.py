"""
Phase 3: Data Extractor — pulls structured data from pages.
Supports TEXT_ONLY and SINGLE_IMAGE modes.
"""
from pydantic import BaseModel
from typing import Any, Type, Optional
from ..llm.llm_gateway import LLMGateway
from ..llm.binary_llm import BinaryLLM
from ..llm.prompt_registry import Prompts
from ..utils.image_processor import ImageProcessor


class DataExtractor:
    def __init__(self, llm_gateway: LLMGateway, binary_llm: BinaryLLM):
        self.llm = llm_gateway
        self.binary = binary_llm
        self.img_proc = ImageProcessor()

    async def extract_text_mode(
        self, full_page_text: str, target_query: str, schema_class: Type[BaseModel]
    ) -> list[Any]:
        """
        Mode A: Extract all items from page text.
        Uses the full page text (not viewport-limited).
        """
        if not full_page_text.strip():
            return []

        return await self.llm.extract_data(full_page_text, target_query, schema_class)

    async def extract_image_mode(
        self,
        page,  # Playwright page
        full_page_text: str,
        target_query: str,
        schema_class: Type[BaseModel],
    ) -> list[Any]:
        """
        Mode B: Single-image data collection.
        1. Find <img> tags in main content
        2. For each, take context bounding box screenshot
        3. Filter relevant images via binary LLM
        4. Extract data from relevant images via main LLM
        """
        results = []

        try:
            # Find all images on the page
            img_elements = await page.query_selector_all("img[src]")

            for img_el in img_elements[:20]:  # Cap at 20 images
                try:
                    src = await img_el.get_attribute("src")
                    if not src or src.startswith("data:image/svg"):
                        continue

                    # Check if image is visible and reasonably sized
                    box = await img_el.bounding_box()
                    if not box or box["width"] < 50 or box["height"] < 50:
                        continue  # Skip tiny icons

                    # Get context screenshot (image + surroundings)
                    parent = img_el
                    try:
                        parent = await img_el.evaluate_handle("el => el.parentElement")
                        context_bytes = await parent.as_element().screenshot(
                            type="jpeg", quality=60
                        )
                    except Exception:
                        context_bytes = await img_el.screenshot(type="jpeg", quality=60)

                    # Get the source image itself
                    source_bytes = await img_el.screenshot(type="jpeg", quality=60)

                    context_b64 = self.img_proc.compress_bytes(context_bytes)
                    source_b64 = self.img_proc.compress_bytes(source_bytes)

                    if not context_b64 or not source_b64:
                        continue

                    # Step 1: Is this image relevant? (Binary LLM)
                    system = Prompts.IMAGE_IMPORTANCE_SYSTEM.format(
                        target_query=target_query
                    )
                    is_relevant = await self.binary.is_image_relevant(
                        context_b64, source_b64, system_prompt=system
                    )

                    if not is_relevant:
                        continue

                    # Step 2: Extract data from relevant image (Main LLM)
                    item = await self.llm.extract_image_data(
                        source_b64, context_b64, full_page_text, schema_class
                    )
                    if item:
                        results.append(item)

                except Exception as e:
                    print(f"[Extractor] Image processing error: {e}")
                    continue

        except Exception as e:
            print(f"[Extractor] Image mode failed: {e}")

        return results
