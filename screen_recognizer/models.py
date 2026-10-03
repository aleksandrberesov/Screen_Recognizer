"""Data models for screen elements, bounding boxes, and screen state."""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


class ElementType(str, Enum):
    BUTTON = "button"
    INPUT = "input"
    TEXT = "text"
    ICON = "icon"
    CHECKBOX = "checkbox"
    CONTAINER = "container"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass
class BoundingBox:
    """Axis-aligned rectangle in pixel coordinates (left, top, right, bottom)."""
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        # Normalize so x1 <= x2 and y1 <= y2
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    @classmethod
    def from_ltwh(cls, x: int, y: int, w: int, h: int) -> BoundingBox:
        """Create from (left, top, width, height)."""
        return cls(int(x), int(y), int(x + w), int(y + h))

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> BoundingBox:
        """Create from (left, top, right, bottom)."""
        return cls(int(left), int(top), int(right), int(bottom))

    @classmethod
    def from_polygon(cls, points: Sequence[Sequence[float]]) -> BoundingBox:
        """Create bounding box enveloping a sequence of [x, y] polygon points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center_x(self) -> int:
        return int((self.x1 + self.x2) / 2)

    @property
    def center_y(self) -> int:
        return int((self.y1 + self.y2) / 2)

    @property
    def center(self) -> tuple[int, int]:
        """Return center coordinate (x, y), ready for mouse click/interaction."""
        return (self.center_x, self.center_y)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width / Height. Returns 0 if height is 0."""
        return (self.width / self.height) if self.height > 0 else 0.0

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    def contains(self, other: BoundingBox, margin: int = 0) -> bool:
        """Check if other is completely contained within this box."""
        return (
            other.x1 >= self.x1 - margin
            and other.y1 >= self.y1 - margin
            and other.x2 <= self.x2 + margin
            and other.y2 <= self.y2 + margin
        )

    def is_inside(self, other: BoundingBox, margin: int = 0) -> bool:
        """Check if this box is completely contained within other."""
        return other.contains(self, margin=margin)

    def intersects(self, other: BoundingBox) -> bool:
        """Check if two boxes overlap at all."""
        return not (
            self.x2 < other.x1
            or self.x1 > other.x2
            or self.y2 < other.y1
            or self.y1 > other.y2
        )

    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        """Calculate the intersection rectangle, or None if no overlap."""
        if not self.intersects(other):
            return None
        return BoundingBox(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union (IoU) metric (0.0 to 1.0)."""
        inter = self.intersection(other)
        if not inter:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def distance_to(self, other: BoundingBox) -> float:
        """Euclidean distance between box centers."""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return math.hypot(dx, dy)


@dataclass
class UIElement:
    """A recognized UI component on the screen."""
    id: int
    type: ElementType
    text: str = ""
    bbox: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 0, 0))
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center

    @property
    def center_x(self) -> int:
        return self.bbox.center_x

    @property
    def center_y(self) -> int:
        return self.bbox.center_y

    @property
    def is_interactive(self) -> bool:
        return self.type in (
            ElementType.BUTTON,
            ElementType.INPUT,
            ElementType.ICON,
            ElementType.CHECKBOX,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, ElementType) else str(self.type),
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "center": [self.center_x, self.center_y],
            "confidence": round(self.confidence, 3),
            "attributes": self.attributes,
        }

    def to_compact_str(self) -> str:
        text_str = f" \"{self.text}\"" if self.text else ""
        return f"[{self.id}] {self.type.value.capitalize()}{text_str} at ({self.center_x}, {self.center_y}) box={self.bbox.to_tuple()}"


@dataclass
class ScreenState:
    """Complete parsed state of a screen or window."""
    elements: list[UIElement] = field(default_factory=list)
    screen_size: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.screen_size[0]

    @property
    def height(self) -> int:
        return self.screen_size[1]

    def find_by_id(self, element_id: int) -> Optional[UIElement]:
        """Find element by its assigned integer ID."""
        for el in self.elements:
            if el.id == element_id:
                return el
        return None

    def find_by_text(
        self,
        query: str,
        exact: bool = False,
        case_sensitive: bool = False,
        min_score: float = 0.6,
    ) -> list[UIElement]:
        """
        Search for elements containing or matching text.
        Supports exact match, substring match, and fuzzy string similarity.
        """
        q = query.strip() if case_sensitive else query.strip().lower()
        if not q:
            return []

        exact_matches = []
        substring_matches = []
        fuzzy_matches: list[tuple[float, UIElement]] = []

        for el in self.elements:
            t = el.text if case_sensitive else el.text.lower()
            if not t:
                continue

            if t == q:
                exact_matches.append(el)
            elif not exact and q in t:
                substring_matches.append(el)
            elif not exact:
                ratio = difflib.SequenceMatcher(None, q, t).ratio()
                if ratio >= min_score:
                    fuzzy_matches.append((ratio, el))

        if exact_matches:
            return exact_matches
        if substring_matches:
            return substring_matches

        # Sort fuzzy by highest similarity ratio
        fuzzy_matches.sort(key=lambda item: -item[0])
        return [el for _, el in fuzzy_matches]

    def find_by_type(self, elem_type: ElementType | str) -> list[UIElement]:
        """Find all elements of a specific type (e.g. 'button', 'input')."""
        target = elem_type.value if isinstance(elem_type, ElementType) else str(elem_type).lower()
        return [el for el in self.elements if el.type.value == target]

    def get_clickable(self) -> list[UIElement]:
        """Get all interactive elements (buttons, inputs, icons, checkboxes)."""
        return [el for el in self.elements if el.is_interactive]

    def find_closest(
        self,
        x: int,
        y: int,
        elem_type: Optional[ElementType | str] = None,
    ) -> Optional[UIElement]:
        """Find element nearest to pixel coordinate (x, y)."""
        candidates = self.elements
        if elem_type:
            target = elem_type.value if isinstance(elem_type, ElementType) else str(elem_type).lower()
            candidates = [el for el in candidates if el.type.value == target]

        if not candidates:
            return None

        return min(candidates, key=lambda el: math.hypot(el.center_x - x, el.center_y - y))

    def to_dict(self) -> dict[str, Any]:
        """Convert full state to standard Python dictionary."""
        return {
            "screen_size": list(self.screen_size),
            "elements_count": len(self.elements),
            "elements": [el.to_dict() for el in self.elements],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize state to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(
        self,
        include_coords: bool = True,
        only_interactive: bool = False,
    ) -> str:
        """
        Produce a compact, token-optimized textual description designed for LLM prompts.
        """
        targets = self.get_clickable() if only_interactive else self.elements
        lines = [
            f"Screen Resolution: {self.width}x{self.height}",
            f"Total Elements: {len(targets)}",
            "",
            "| ID | Type | Text / Content | Center (X, Y) | Bounding Box [x1, y1, x2, y2] |",
            "|---|---|---|---|---|",
        ]
        for el in targets:
            text_disp = el.text.replace("|", "\\|").replace("\n", " ").strip()
            if not text_disp:
                text_disp = "-"
            coords = f"({el.center_x}, {el.center_y})" if include_coords else "-"
            bbox = f"[{el.bbox.x1}, {el.bbox.y1}, {el.bbox.x2}, {el.bbox.y2}]" if include_coords else "-"
            lines.append(f"| [{el.id}] | {el.type.value} | {text_disp} | {coords} | {bbox} |")

        return "\n".join(lines)
