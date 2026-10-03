"""Local Computer Vision UI component and icon detector using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .models import BoundingBox, ElementType


@dataclass
class DetectedBox:
    """A raw visual contour/box detected by computer vision."""
    bbox: BoundingBox
    category: ElementType
    confidence: float
    hierarchy_depth: int = 0
    contour_area: float = 0.0
    # Outline metrics used for shape classification in reports
    vertex_count: int = 0
    polygon_area: float = 0.0
    hull_area: float = 0.0
    perimeter: float = 0.0


class UIElementDetector:
    """
    Fast local CV detector that identifies rectangular buttons, text inputs,
    checkboxes, and icons without requiring neural network weights.
    """

    def __init__(
        self,
        min_element_size: int = 12,
        max_element_ratio: float = 0.95,
    ):
        """
        :param min_element_size: Minimum width and height (in px) for an element.
        :param max_element_ratio: Maximum fraction of total screen area an element can occupy.
        """
        self.min_element_size = min_element_size
        self.max_element_ratio = max_element_ratio

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        """
        Detect UI element boundaries on the screen image.
        Returns a list of DetectedBox objects.
        """
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        screen_area = h * w
        max_area = screen_area * self.max_element_ratio
        min_area = self.min_element_size * self.min_element_size

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Multi-scale edge / gradient extraction
        # 1. Morphological gradient captures sharp UI button edges & borders
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_rect)

        # 2. Canny edges with Otsu-based automatic thresholding
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny = cv2.Canny(gray, otsu_thresh * 0.5, otsu_thresh)

        # 3. Combine gradient & canny
        combined = cv2.bitwise_or(gradient, canny)

        # 4. Connect dashed/subtle UI borders
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_rect, iterations=1)

        # Find contours with full tree hierarchy
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            return []

        hierarchy = hierarchy[0]
        detected: list[DetectedBox] = []

        for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            # Approximate contour to polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            # Filter out invalid dimensions
            if bw < self.min_element_size or bh < self.min_element_size:
                continue
            if area < min_area or area > max_area:
                continue

            # Check aspect ratio
            aspect = bw / float(bh) if bh > 0 else 0.0
            if aspect > 25.0 or aspect < 0.04:
                continue

            # Determine depth in hierarchy
            parent_idx = hier[3]
            depth = 0
            curr_parent = parent_idx
            while curr_parent != -1 and depth < 10:
                depth += 1
                curr_parent = hierarchy[curr_parent][3]

            bbox = BoundingBox.from_ltwh(x, y, bw, bh)
            category = self._classify_shape(bw, bh, aspect, len(approx), screen_width=w, screen_height=h)

            detected.append(
                DetectedBox(
                    bbox=bbox,
                    category=category,
                    confidence=0.85,
                    hierarchy_depth=depth,
                    contour_area=float(area),
                    vertex_count=len(approx),
                    polygon_area=float(cv2.contourArea(cnt)),
                    hull_area=float(cv2.contourArea(cv2.convexHull(cnt))),
                    perimeter=float(peri),
                )
            )

        # Filter nested duplicates (keep most informative components)
        return self._filter_duplicates(detected)

    def _classify_shape(
        self,
        width: int,
        height: int,
        aspect: float,
        num_vertices: int,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> ElementType:
        """Classify visual shape into a UI element category based on dimensions."""
        # Top-level banners or wide nav bars
        if width >= int(screen_width * 0.75) or height >= int(screen_height * 0.75):
            return ElementType.CONTAINER

        # Small square or small circle -> Checkbox / Radio
        if 12 <= width <= 36 and 12 <= height <= 36 and 0.8 <= aspect <= 1.25:
            return ElementType.CHECKBOX

        # Small icon / tool button
        if 14 <= width <= 64 and 14 <= height <= 64 and 0.7 <= aspect <= 1.4:
            return ElementType.ICON

        # Standard button
        if 35 <= width <= 400 and 18 <= height <= 65 and 1.1 <= aspect <= 8.0:
            return ElementType.BUTTON

        # Horizontal input field
        if 50 <= width <= 600 and 18 <= height <= 55 and aspect >= 2.5:
            return ElementType.INPUT

        # Larger block
        if width > 120 and height > 80:
            return ElementType.CONTAINER

        return ElementType.UNKNOWN

    def _filter_duplicates(self, boxes: list[DetectedBox]) -> list[DetectedBox]:
        """Apply non-maximum suppression / deduplication for concentric contour borders."""
        if not boxes:
            return []

        # Sort by area descending so outer borders come first
        boxes.sort(key=lambda b: -b.contour_area)

        kept: list[DetectedBox] = []
        for box in boxes:
            duplicate = False
            for existing in kept:
                # 1. High IoU overlap
                if box.bbox.iou(existing.bbox) > 0.70:
                    duplicate = True
                    break
                # 2. Concentric box (border stroke width artifact: box is inside existing with small margin)
                if existing.bbox.contains(box.bbox, margin=2):
                    dw = existing.bbox.width - box.bbox.width
                    dh = existing.bbox.height - box.bbox.height
                    if dw <= 8 and dh <= 8:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(box)

        return kept
