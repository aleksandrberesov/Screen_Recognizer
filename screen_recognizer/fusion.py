"""Spatial fusion engine: reconciles OCR text and CV detected boxes into unified UI elements."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .detector import DetectedBox
from .models import BoundingBox, ElementType, ScreenState, UIElement
from .ocr_engine import OCRResult


# Common button keywords and action verbs
_ACTION_WORDS = {
    "ok", "cancel", "apply", "save", "submit", "close", "yes", "no",
    "login", "sign in", "sign up", "register", "search", "back", "next",
    "finish", "edit", "delete", "copy", "paste", "continue", "download",
    "deploy", "run", "start", "stop", "test", "confirm", "reset", "send",
    "add", "remove", "update", "upload", "view", "help", "settings"
}


def is_container_box(cv_box: DetectedBox) -> bool:
    """Whether a detected box is large enough to be a card/panel rather than a single control."""
    return (cv_box.bbox.width > 250 and cv_box.bbox.height > 80) or cv_box.category == ElementType.CONTAINER


class SpatialFusionEngine:
    """
    Fuses local OCR text detections with CV visual contours to classify and ground
    actionable UI components (buttons, input boxes, icons, labels).
    """

    def __init__(
        self,
        box_containment_margin: int = 4,
        max_button_text_length: int = 50,
        include_containers: bool = False,
    ):
        self.box_containment_margin = box_containment_margin
        self.max_button_text_length = max_button_text_length
        self.include_containers = include_containers

    def fuse(
        self,
        ocr_results: list[OCRResult],
        cv_boxes: list[DetectedBox],
        screen_size: tuple[int, int],
    ) -> ScreenState:
        """
        Merge OCR results and CV detections into a unified ScreenState.
        """
        raw_elements: list[UIElement] = []
        matched_ocr_indices = set()
        matched_cv_indices = set()

        # Step 1: Match OCR text with CV bounding boxes (Buttons / Labeled boxes)
        for cv_idx, cv_box in enumerate(cv_boxes):
            # Containers are added after deduplication so they never claim the text of the controls inside them
            if is_container_box(cv_box):
                continue

            contained_texts: list[OCRResult] = []
            for ocr_idx, ocr in enumerate(ocr_results):
                # Check if OCR box is inside the CV box or significantly overlaps
                if cv_box.bbox.contains(ocr.bbox, margin=self.box_containment_margin) or cv_box.bbox.iou(ocr.bbox) > 0.4:
                    contained_texts.append(ocr)
                    matched_ocr_indices.add(ocr_idx)

            if contained_texts:
                matched_cv_indices.add(cv_idx)
                contained_texts.sort(key=lambda o: (o.bbox.y1 // 15, o.bbox.x1))
                full_text = " ".join(o.text for o in contained_texts).strip()
                avg_conf = sum(o.confidence for o in contained_texts) / len(contained_texts)

                # Determine element type based on text content & box geometry
                elem_type = self._resolve_element_type(cv_box, full_text)

                raw_elements.append(
                    UIElement(
                        id=0,
                        type=elem_type,
                        text=full_text,
                        bbox=cv_box.bbox,
                        confidence=round((avg_conf + cv_box.confidence) / 2.0, 3),
                    )
                )

        # Step 2: Handle unmatched CV boxes (e.g. Empty Input fields, Checkboxes, Icons)
        for cv_idx, cv_box in enumerate(cv_boxes):
            if cv_idx in matched_cv_indices:
                continue

            if is_container_box(cv_box):
                continue

            if cv_box.category == ElementType.INPUT:
                label = self._find_nearest_label(cv_box.bbox, ocr_results)
                raw_elements.append(
                    UIElement(
                        id=0,
                        type=ElementType.INPUT,
                        text=label or "",
                        bbox=cv_box.bbox,
                        confidence=cv_box.confidence,
                        attributes={"label": label or ""},
                    )
                )
            elif cv_box.category in (ElementType.CHECKBOX, ElementType.ICON):
                label = self._find_nearest_label(cv_box.bbox, ocr_results, max_dist=100)
                raw_elements.append(
                    UIElement(
                        id=0,
                        type=cv_box.category,
                        text=label or "",
                        bbox=cv_box.bbox,
                        confidence=cv_box.confidence,
                    )
                )

        # Step 3: Handle unmatched OCR text (Standalone text, Flat buttons, Hyperlinks)
        for ocr_idx, ocr in enumerate(ocr_results):
            if ocr_idx in matched_ocr_indices:
                continue

            txt = ocr.text.strip()
            # If text words match common action words, classify as BUTTON
            words = set(re.findall(r"\b\w+\b", txt.lower()))
            is_action = bool(words & _ACTION_WORDS) and len(txt) <= 30
            elem_type = ElementType.BUTTON if is_action else ElementType.TEXT

            raw_elements.append(
                UIElement(
                    id=0,
                    type=elem_type,
                    text=txt,
                    bbox=ocr.bbox,
                    confidence=ocr.confidence,
                )
            )

        # Step 4: Final Deduplication & Non-Maximum Suppression
        elements = self._deduplicate_elements(raw_elements)

        # Containers enclose other elements by design, so they are deduplicated only among themselves
        if self.include_containers:
            elements.extend(self._deduplicate_containers([
                UIElement(
                    id=0,
                    type=ElementType.CONTAINER,
                    text="",
                    bbox=cv_box.bbox,
                    confidence=cv_box.confidence,
                )
                for cv_box in cv_boxes
                if is_container_box(cv_box)
            ]))

        # Step 5: Sort elements by natural reading order (row bucketed by 25px, then x1)
        elements.sort(key=lambda e: (e.bbox.y1 // 25, e.bbox.x1))

        # Assign sequential IDs
        for idx, elem in enumerate(elements):
            elem.id = idx

        return ScreenState(
            elements=elements,
            screen_size=screen_size,
            metadata={"fused_elements_count": len(elements)},
        )

    def _resolve_element_type(self, cv_box: DetectedBox, text: str) -> ElementType:
        """Heuristic to decide between BUTTON, INPUT, and TEXT."""
        txt_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", txt_lower))

        # Check for input characteristics (email, url, placeholder patterns)
        if "@" in text or "http" in txt_lower or "search" in txt_lower or cv_box.bbox.width > 350:
            if "search" in txt_lower:
                return ElementType.INPUT
            if "@" in text or "password" in txt_lower:
                return ElementType.INPUT

        # Action verbs or short labels in standard button dimensions -> BUTTON
        if bool(words & _ACTION_WORDS) or (len(text) <= self.max_button_text_length and cv_box.bbox.height <= 60):
            return ElementType.BUTTON

        if cv_box.category != ElementType.UNKNOWN:
            return cv_box.category

        return ElementType.TEXT

    @staticmethod
    def _deduplicate_containers(containers: list[UIElement]) -> list[UIElement]:
        """Drop near-identical container boxes while keeping nested ones (a card inside a panel)."""
        kept: list[UIElement] = []
        for container in sorted(containers, key=lambda c: -c.bbox.area):
            if all(container.bbox.iou(existing.bbox) <= 0.7 for existing in kept):
                kept.append(container)
        return kept

    def _deduplicate_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Eliminate duplicate or concentric elements covering the same UI control."""
        if not elements:
            return []

        # Sort elements so more specific/interactive elements come first
        def priority(e: UIElement) -> int:
            p = 0
            if e.is_interactive:
                p += 10
            if e.type == ElementType.BUTTON:
                p += 5
            if e.type == ElementType.INPUT:
                p += 4
            if e.text:
                p += 2
            return -p  # Highest priority first

        elements.sort(key=priority)

        kept: list[UIElement] = []
        for el in elements:
            duplicate = False
            for idx, existing in enumerate(kept):
                # 1. Same region (IoU > 0.45 or center distance < 20)
                if el.bbox.iou(existing.bbox) > 0.45 or el.bbox.distance_to(existing.bbox) < 20:
                    duplicate = True
                    break

                # 2. Concentric box: if existing already encloses el, el is an inner duplicate
                if existing.bbox.contains(el.bbox, margin=5):
                    duplicate = True
                    break

                # 3. If el encloses existing, el is the true outer control box (replace existing)
                if el.bbox.contains(existing.bbox, margin=5):
                    kept[idx] = el
                    duplicate = True
                    break

            if not duplicate:
                kept.append(el)

        return kept

    @staticmethod
    def _find_nearest_label(
        target_box: BoundingBox,
        ocr_results: list[OCRResult],
        max_dist: int = 150,
    ) -> Optional[str]:
        """Find an OCR text label directly to the left or above the target box."""
        candidates: list[tuple[float, str]] = []
        for ocr in ocr_results:
            # Check left-side label (same vertical level)
            y_overlap = min(target_box.y2, ocr.bbox.y2) - max(target_box.y1, ocr.bbox.y1)
            if y_overlap > 5 and ocr.bbox.x2 <= target_box.x1:
                dist = target_box.x1 - ocr.bbox.x2
                if dist <= max_dist:
                    candidates.append((dist, ocr.text))

            # Check above label (aligned horizontally)
            x_overlap = min(target_box.x2, ocr.bbox.x2) - max(target_box.x1, ocr.bbox.x1)
            if x_overlap > 5 and ocr.bbox.y2 <= target_box.y1:
                dist = target_box.y1 - ocr.bbox.y2
                if dist <= 50:
                    candidates.append((dist * 1.5, ocr.text))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates[0][1]

        return None
