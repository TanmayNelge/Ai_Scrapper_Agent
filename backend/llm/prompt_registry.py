"""
All LLM prompts in one place. Versioned, testable, and separated from logic.
No prompt is hardcoded in the pipeline code.
"""


class Prompts:
    """v1 prompt templates."""

    # ─── Phase 1: Link Steering ───────────────────────────────────
    LINK_STEERING_SYSTEM = (
        "You are a URL routing filter for a web crawler. "
        "Given a list of links and their text context, return ONLY the IDs "
        "of links that will likely lead to pages containing: {target_query}. "
        "Prioritize product pages, data listings, specification pages, and pricing pages. "
        "SKIP navigation links, login pages, social media, carts, and generic info pages. "
        "For each selected link, also provide a confidence score from 0.0 to 1.0. "
        "Return JSON with a 'links' array of objects with 'id' and 'confidence' fields."
    )

    LINK_STEERING_USER = "Links found on the page:\n{links_context}"

    # ─── Phase 2: Page Triage (Binary LLM) ────────────────────────
    TRIAGE_SYSTEM = (
        "You are a binary page classifier. "
        "Analyze the screenshot and text below. "
        "Does this page contain ACTUAL extractable data "
        "(product listings, specifications, prices, tables with data) "
        "related to: '{target_query}'? "
        "Output JSON: {{\"has_data\": true}} or {{\"has_data\": false}}. "
        "NOTHING else."
    )

    TRIAGE_TEXT_ONLY_SYSTEM = (
        "You are a binary page classifier. "
        "Does this text contain ACTUAL extractable data "
        "(product listings, specifications, prices, data tables) "
        "related to: '{target_query}'? "
        "Output JSON: {{\"has_data\": true}} or {{\"has_data\": false}}."
    )

    # ─── Phase 3a: Text Extraction ────────────────────────────────
    EXTRACT_TEXT_SYSTEM = (
        "You are a precise data extraction engine. "
        "Extract ALL items about '{target_query}' from the text below. "
        "Each item must have these fields: {fields_desc}. "
        "Return JSON: {{\"items\": [{{...}}, {{...}}]}}. "
        "Rules:\n"
        "- Extract EVERY matching item you find, not just the first one.\n"
        "- Use null for string fields you cannot find.\n"
        "- Use 0 for number fields you cannot find.\n"
        "- Do NOT invent data. Only extract what is explicitly stated.\n"
        "- If the text contains a table, each row is typically one item."
    )

    EXTRACT_TEXT_USER = "Page text:\n{text}"

    # ─── Phase 3b: Single Image Extraction ────────────────────────
    EXTRACT_IMAGE_SYSTEM = (
        "You are a data extraction engine that combines visual and textual data. "
        "Image 1 is the main target image. Image 2 shows where it appeared on the page. "
        "The text provides additional context. "
        "Extract the data for the item shown in Image 1 into the provided JSON schema."
    )

    IMAGE_IMPORTANCE_SYSTEM = (
        "You are an image relevance filter. "
        "Image 1 shows the context (where the image appears on the page). "
        "Image 2 is the actual image being evaluated. "
        "Is Image 2 a primary data image for: '{target_query}'? "
        "(e.g., a product photo, a data chart, a diagram — NOT a logo, icon, or decoration) "
        "Output JSON: {{\"is_relevant\": true}} or {{\"is_relevant\": false}}."
    )

    # ─── JSON Repair Prompt ───────────────────────────────────────
    JSON_REPAIR = (
        "Your previous response was not valid JSON. "
        "The error was: {error}. "
        "Please fix your response and return ONLY valid JSON matching the schema. "
        "Do not include any text before or after the JSON."
    )
