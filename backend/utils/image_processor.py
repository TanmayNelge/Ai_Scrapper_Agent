"""
Image processor for compressing screenshots and product images
before sending to the Vision LLM. Uses WhatsApp-style compression.
"""
import base64
import io
from PIL import Image
from typing import Optional


class ImageProcessor:
    def __init__(self, max_dimension: int = 1024, jpeg_quality: int = 60):
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def compress_bytes(self, image_bytes: bytes) -> Optional[str]:
        """
        Compress raw image bytes → base64 string for LLM.
        Returns None on failure instead of empty string.
        """
        if not image_bytes:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB (handles RGBA, P, LA modes)
            if img.mode not in ("RGB",):
                img = img.convert("RGB")

            # Resize if exceeds max_dimension (preserve aspect ratio)
            width, height = img.size
            if width > self.max_dimension or height > self.max_dimension:
                if width >= height:
                    new_w = self.max_dimension
                    new_h = int(height * (self.max_dimension / width))
                else:
                    new_h = self.max_dimension
                    new_w = int(width * (self.max_dimension / height))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"[ImageProcessor] Compression failed: {e}")
            return None

    def screenshot_to_b64(self, screenshot_bytes: bytes) -> Optional[str]:
        """Convenience alias for viewport screenshots."""
        return self.compress_bytes(screenshot_bytes)
