"""Set-of-Mark (SoM) visual annotator for drawing visual grounding badges on screenshots."""

from __future__ import annotations

from typing import Optional, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .models import ElementType, ScreenState, UIElement

# Distinct high-visibility colors for element categories
_TYPE_COLORS = {
    ElementType.BUTTON: "#22c55e",     # Green
    ElementType.INPUT: "#3b82f6",      # Blue
    ElementType.TEXT: "#eab308",       # Yellow
    ElementType.ICON: "#ec4899",       # Pink
    ElementType.CHECKBOX: "#8b5cf6",   # Purple
    ElementType.CONTAINER: "#64748b",  # Slate
    ElementType.UNKNOWN: "#f97316",    # Orange
}


class VisualAnnotator:
    """Renders Set-of-Mark visual grounding tags and bounding boxes on screenshots."""

    def __init__(self, font_size: int = 14):
        self.font_size = font_size
        self._font = None

    def _get_font(self):
        if self._font is None:
            try:
                # Attempt to load Arial or Segoe UI on Windows
                self._font = ImageFont.truetype("arial.ttf", self.font_size)
            except Exception:
                try:
                    self._font = ImageFont.truetype("segoeui.ttf", self.font_size)
                except Exception:
                    self._font = ImageFont.load_default()
        return self._font

    def annotate(
        self,
        image: Image.Image,
        state: ScreenState,
        output_path: Optional[str] = None,
        only_interactive: bool = False,
        draw_boxes: bool = True,
        draw_labels: bool = True,
    ) -> Image.Image:
        """
        Draw bounding boxes and numbered ID badges [0], [1], [2]... on the image.
        
        :param image: Original PIL Image.
        :param state: Parsed ScreenState containing UIElements.
        :param output_path: Optional path to save the annotated PNG/JPEG.
        :param only_interactive: If True, only draws badges on clickable/input elements.
        :param draw_boxes: Whether to draw bounding rectangle outlines.
        :param draw_labels: Whether to draw the numeric badge [ID].
        """
        annotated = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._get_font()

        targets = state.get_clickable() if only_interactive else state.elements

        for el in targets:
            color_hex = _TYPE_COLORS.get(el.type, "#22c55e")
            box = el.bbox

            # Draw semi-transparent fill and border
            if draw_boxes:
                # Border
                draw.rectangle(
                    [box.x1, box.y1, box.x2, box.y2],
                    outline=color_hex,
                    width=2,
                )

            if draw_labels:
                badge_text = f"{el.id}"
                # Measure text size
                try:
                    bbox = font.getbbox(badge_text)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = len(badge_text) * 8, 12

                pad = 3
                # Badge position (top-left corner of the element, or offset if near top)
                bx1 = max(0, box.x1)
                by1 = max(0, box.y1 - th - pad * 2)
                if by1 < 0:
                    by1 = box.y1

                bx2 = bx1 + tw + pad * 2
                by2 = by1 + th + pad * 2

                # Solid badge background with dark outline
                draw.rectangle([bx1, by1, bx2, by2], fill=color_hex, outline="#000000", width=1)
                # Black badge text
                draw.text((bx1 + pad, by1 + pad - 1), badge_text, fill="#000000", font=font)

        result = Image.alpha_composite(annotated, overlay).convert("RGB")

        if output_path:
            result.save(output_path)

        return result
