"""Full screen report: every element with its hierarchy, geometry, shape, colors and relations.

The report is a self-describing JSON document for handing screen state to other tools.
Its JSON Schema comes from ``ScreenReport.model_json_schema()``.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .detector import DetectedBox
from .fusion import is_container_box
from .models import BoundingBox, ElementType, ScreenState, UIElement
from .ocr_engine import OCRResult

REPORT_FORMAT = "screen-recognizer.report"
REPORT_SCHEMA_VERSION = "1.0"

ShapeKind = Literal["rectangle", "rounded_rectangle", "ellipse", "polygon", "irregular", "text_region"]

_LABELLED_TYPES = (ElementType.INPUT, ElementType.CHECKBOX, ElementType.ICON)
_TIMING_ORDER = ("capture", "ocr", "detection", "fusion")


# --------------------------------------------------------------------------- #
# Report models
# --------------------------------------------------------------------------- #

class Point(BaseModel):
    x: int
    y: int


class Box(BaseModel):
    """Axis-aligned rectangle in image pixels: (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def of(cls, bbox: BoundingBox) -> Box:
        return cls(**bbox.to_dict())


class Geometry(BaseModel):
    width: int
    height: int
    area: int
    aspect_ratio: float = Field(description="width / height")
    screen_coverage: float = Field(description="Fraction of the image area covered by the element (0-1)")
    touches_edge: bool = Field(
        description="Box touches the image border, so the element may be cut off and its shape metrics unreliable"
    )


class Shape(BaseModel):
    kind: ShapeKind = Field(
        description="Outline shape. 'text_region' means no drawn outline was analyzed (OCR-only or plain text)"
    )
    vertex_count: Optional[int] = Field(None, description="Corners of the simplified outline polygon")
    extent: Optional[float] = Field(
        None, description="Outline area / bounding box area; ~1.0 for rectangles, ~0.785 for ellipses"
    )
    solidity: Optional[float] = Field(
        None, description="Outline area / convex hull area; low values mean a concave or broken outline"
    )
    circularity: Optional[float] = Field(
        None, description="4*pi*area / perimeter^2; 1.0 for a circle, ~0.785 for a square"
    )
    has_border: bool = Field(description="A visible border differs in color from the element background")


class Style(BaseModel):
    background_color: str = Field(description="Dominant fill color, #rrggbb")
    border_color: Optional[str] = Field(None, description="Border color, #rrggbb; null when no visible border")
    text_color: Optional[str] = Field(
        None, description="Estimated text color, #rrggbb; null when the element has no OCR text inside it"
    )
    brightness: float = Field(description="Mean pixel intensity of the element region (0 = black, 1 = white)")
    contrast_ratio: Optional[float] = Field(
        None, description="WCAG contrast ratio of text against background (1-21)"
    )


class TextFragment(BaseModel):
    text: str
    bbox: Box
    confidence: float


class Hierarchy(BaseModel):
    parent_id: Optional[int] = Field(
        None, description="Smallest container enclosing this element; null for top-level elements"
    )
    children_ids: list[int] = Field(default_factory=list, description="Direct children in reading order")
    depth: int = Field(description="0 for top-level elements")
    path: list[int] = Field(
        default_factory=list, description="Ancestor IDs from the top-level container down to the parent"
    )


class Layout(BaseModel):
    row: int = Field(description="Visual row among siblings (same parent), top to bottom from 0")
    column: int = Field(description="Position within its row, left to right from 0")
    region: str = Field(description="Screen third the center falls in, e.g. 'top-left', 'center', 'bottom-right'")


class Neighbors(BaseModel):
    """Nearest element in each direction that overlaps on the other axis (containers only neighbor containers)."""
    left: Optional[int] = None
    right: Optional[int] = None
    above: Optional[int] = None
    below: Optional[int] = None


class Relations(BaseModel):
    neighbors: Neighbors
    labelled_by: Optional[int] = Field(None, description="Text element labelling this input, checkbox or icon")
    label_for: list[int] = Field(default_factory=list, description="Elements this text element labels")


class Provenance(BaseModel):
    sources: list[Literal["ocr", "cv"]] = Field(
        description="Detectors that contributed: OCR text, a CV outline, or both"
    )
    cv_category: Optional[str] = Field(None, description="Category the shape detector guessed before fusion")
    contour_depth: Optional[int] = Field(None, description="Nesting depth of the outline in the contour tree")


class ReportElement(BaseModel):
    id: int = Field(
        description="Unique within the report. Elements returned by ScreenRecognizer.recognize() keep their IDs "
        "(so to_prompt() IDs work here); containers and shapes it did not return are numbered after them"
    )
    type: ElementType
    text: str
    interactive: bool
    confidence: float
    bbox: Box
    center: Point = Field(description="Center in image pixels")
    screen_center: Point = Field(
        description="Center in absolute screen pixels (center + source.offset); use this to click"
    )
    geometry: Geometry
    shape: Shape
    style: Style
    text_fragments: list[TextFragment] = Field(
        default_factory=list, description="Raw OCR lines merged into this element"
    )
    hierarchy: Hierarchy
    layout: Layout
    relations: Relations
    provenance: Provenance
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Extra facts, e.g. 'title' of a card or 'label' of an empty input"
    )


class TreeNode(BaseModel):
    id: int
    type: ElementType
    text: str = Field(description="Element text, or the title for containers")
    children: list[TreeNode] = Field(default_factory=list)


TreeNode.model_rebuild()


class SourceInfo(BaseModel):
    kind: Literal["screen", "window", "file", "image"]
    path: Optional[str] = None
    window_title: Optional[str] = None
    offset: Point = Field(
        default_factory=lambda: Point(x=0, y=0),
        description="Absolute screen position of image pixel (0, 0)",
    )


class ColorShare(BaseModel):
    color: str
    share: float = Field(description="Fraction of pixels (0-1)")


class ImageInfo(BaseModel):
    width: int
    height: int
    mean_color: str
    dominant_colors: list[ColorShare]


class ToolInfo(BaseModel):
    name: str = "screen-recognizer"
    version: str
    ocr_backend: str
    cv_detection: bool


class Summary(BaseModel):
    total_elements: int
    interactive_elements: int
    by_type: dict[str, int]
    max_depth: int
    ocr_fragments: int = Field(description="Raw OCR text lines before fusion")
    cv_shapes: int = Field(description="Raw CV outlines before fusion")


class RawOCR(BaseModel):
    text: str
    bbox: Box
    confidence: float
    element_id: Optional[int] = Field(None, description="Element this text ended up in; null if it was dropped")


class RawShape(BaseModel):
    bbox: Box
    category: str
    contour_depth: int
    vertex_count: int
    element_id: Optional[int] = Field(
        None, description="Element built from this outline; null if it was merged into another element or discarded"
    )


class RawDetections(BaseModel):
    ocr: list[RawOCR]
    cv: list[RawShape]


class ScreenReport(BaseModel):
    """Structured description of one screenshot: every element with its hierarchy, geometry, shape, colors and relations."""

    report_format: str = REPORT_FORMAT
    schema_version: str = REPORT_SCHEMA_VERSION
    generated_at: str = Field(description="UTC timestamp, ISO 8601")
    tool: ToolInfo
    source: SourceInfo
    image: ImageInfo
    timings_ms: dict[str, float]
    summary: Summary
    elements: list[ReportElement] = Field(description="All elements, sorted by id")
    tree: list[TreeNode] = Field(description="Top-level elements with nested children, in reading order")
    raw: Optional[RawDetections] = Field(None, description="Unfused OCR and CV detections; null when excluded")

    def find_by_id(self, element_id: int) -> Optional[ReportElement]:
        return next((el for el in self.elements if el.id == element_id), None)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    def to_screen_state(self) -> ScreenState:
        """All elements as a plain ScreenState with the same IDs, e.g. for annotate() or find_by_text()."""
        return ScreenState(
            elements=[
                UIElement(
                    id=el.id,
                    type=el.type,
                    text=el.text,
                    bbox=BoundingBox(el.bbox.x1, el.bbox.y1, el.bbox.x2, el.bbox.y2),
                    confidence=el.confidence,
                    attributes=dict(el.attributes),
                )
                for el in self.elements
            ],
            screen_size=(self.image.width, self.image.height),
            metadata={"offset_x": self.source.offset.x, "offset_y": self.source.offset.y},
        )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

@dataclass
class _Node:
    element: UIElement
    shape: Optional[DetectedBox]
    fragments: list[OCRResult]


class ReportBuilder:
    """Turns one recognition pass (image, fused state, raw OCR and CV detections) into a ScreenReport."""

    def __init__(self, containment_margin: int = 4):
        self.containment_margin = containment_margin

    def build(
        self,
        image: Image.Image,
        state: ScreenState,
        ocr_results: list[OCRResult],
        cv_boxes: list[DetectedBox],
        source: SourceInfo,
        tool: ToolInfo,
        timings_ms: Optional[dict[str, float]] = None,
        include_raw: bool = True,
    ) -> ScreenReport:
        t0 = time.perf_counter()
        rgb = np.asarray(image.convert("RGB"))
        width, height = image.size

        nodes = self._collect_nodes(state, ocr_results, cv_boxes)
        by_id = {n.element.id: n for n in nodes}
        boxes = {eid: n.element.bbox for eid, n in by_id.items()}

        # Hierarchy: each element's parent is the smallest container enclosing it
        containers = sorted(
            (n.element for n in nodes if n.element.type == ElementType.CONTAINER),
            key=lambda el: el.bbox.area,
        )
        parent_of: dict[int, Optional[int]] = {}
        for eid, box in boxes.items():
            parent_of[eid] = next(
                (
                    c.id for c in containers
                    if c.id != eid and c.bbox.area > box.area
                    and c.bbox.contains(box, margin=self.containment_margin)
                ),
                None,
            )

        children: dict[Optional[int], list[int]] = defaultdict(list)
        for eid, pid in parent_of.items():
            children[pid].append(eid)
        grid: dict[int, tuple[int, int]] = {}
        for sibling_ids in children.values():
            grid.update(_assign_rows(sibling_ids, boxes))
            sibling_ids.sort(key=lambda i: grid[i])

        # Labels: nearest text beside or above an input/checkbox/icon, within the same parent
        labelled_by: dict[int, int] = {}
        for eid, node in by_id.items():
            if node.element.type in _LABELLED_TYPES:
                texts = [
                    by_id[i].element for i in children.get(parent_of[eid], [])
                    if by_id[i].element.type == ElementType.TEXT
                ]
                label_id = _find_label(node.element, texts)
                if label_id is not None:
                    labelled_by[eid] = label_id
        label_for: dict[int, list[int]] = defaultdict(list)
        for eid, label_id in labelled_by.items():
            label_for[label_id].append(eid)

        # Container title: a text child sitting at the top of a card or panel
        titles: dict[int, str] = {}
        for container in containers:
            if container.bbox.height < 80:
                continue
            for cid in children.get(container.id, []):
                child = by_id[cid].element
                if child.type == ElementType.TEXT and child.bbox.y1 - container.bbox.y1 <= 40:
                    titles[container.id] = child.text
                    break

        container_ids = [c.id for c in containers]
        leaf_ids = [eid for eid in boxes if by_id[eid].element.type != ElementType.CONTAINER]
        ox, oy = source.offset.x, source.offset.y
        screen_area = max(1, width * height)

        elements: list[ReportElement] = []
        for eid in sorted(by_id):
            node = by_id[eid]
            el = node.element
            # An unclassified outline around OCR text is just the glyphs, not a drawn control
            outlined = (
                node.shape is not None
                and el.type != ElementType.TEXT
                and not (node.fragments and node.shape.category == ElementType.UNKNOWN)
            )
            style, has_border = _describe_style(rgb, el.bbox, node.fragments, outlined)
            path = _ancestors(eid, parent_of)
            row, column = grid[eid]
            attributes = dict(el.attributes)
            if eid in titles:
                attributes["title"] = titles[eid]

            elements.append(ReportElement(
                id=eid,
                type=el.type,
                text=el.text,
                interactive=el.is_interactive,
                confidence=round(el.confidence, 3),
                bbox=Box.of(el.bbox),
                center=Point(x=el.center_x, y=el.center_y),
                screen_center=Point(x=el.center_x + ox, y=el.center_y + oy),
                geometry=Geometry(
                    width=el.bbox.width,
                    height=el.bbox.height,
                    area=el.bbox.area,
                    aspect_ratio=round(el.bbox.aspect_ratio, 3),
                    screen_coverage=round(el.bbox.area / screen_area, 5),
                    touches_edge=(
                        el.bbox.x1 <= 0 or el.bbox.y1 <= 0
                        or el.bbox.x2 >= width - 1 or el.bbox.y2 >= height - 1
                    ),
                ),
                shape=_describe_shape(node.shape if outlined else None, has_border),
                style=style,
                text_fragments=[
                    TextFragment(text=f.text, bbox=Box.of(f.bbox), confidence=round(f.confidence, 3))
                    for f in node.fragments
                ],
                hierarchy=Hierarchy(
                    parent_id=parent_of[eid],
                    children_ids=list(children.get(eid, [])),
                    depth=len(path),
                    path=path,
                ),
                layout=Layout(row=row, column=column, region=_region(el.center_x, el.center_y, width, height)),
                relations=Relations(
                    neighbors=_neighbors(eid, container_ids if el.type == ElementType.CONTAINER else leaf_ids, boxes),
                    labelled_by=labelled_by.get(eid),
                    label_for=sorted(label_for.get(eid, [])),
                ),
                provenance=Provenance(
                    sources=(["ocr"] if node.fragments else []) + (["cv"] if node.shape else []),
                    cv_category=node.shape.category.value if node.shape else None,
                    contour_depth=node.shape.hierarchy_depth if node.shape else None,
                ),
                attributes=attributes,
            ))

        def to_tree(eid: int) -> TreeNode:
            el = by_id[eid].element
            return TreeNode(
                id=eid,
                type=el.type,
                text=el.text or titles.get(eid, ""),
                children=[to_tree(cid) for cid in children.get(eid, [])],
            )

        by_type = Counter(el.type.value for el in elements)
        summary = Summary(
            total_elements=len(elements),
            interactive_elements=sum(el.interactive for el in elements),
            by_type=dict(sorted(by_type.items())),
            max_depth=max((el.hierarchy.depth for el in elements), default=0),
            ocr_fragments=len(ocr_results),
            cv_shapes=len(cv_boxes),
        )
        image_info = _describe_image(rgb)
        raw = self._raw_detections(nodes, ocr_results, cv_boxes) if include_raw else None

        timings = {
            k: round(v, 1)
            for k, v in sorted(
                (timings_ms or {}).items(),
                key=lambda kv: _TIMING_ORDER.index(kv[0]) if kv[0] in _TIMING_ORDER else len(_TIMING_ORDER),
            )
        }
        timings["report"] = round((time.perf_counter() - t0) * 1000.0, 1)
        timings["total"] = round(sum(timings.values()), 1)

        return ScreenReport(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tool=tool,
            source=source,
            image=image_info,
            timings_ms=timings,
            summary=summary,
            elements=elements,
            tree=[to_tree(eid) for eid in children.get(None, [])],
            raw=raw,
        )

    def _collect_nodes(
        self,
        state: ScreenState,
        ocr_results: list[OCRResult],
        cv_boxes: list[DetectedBox],
    ) -> list[_Node]:
        """Pair fused elements with their outline and OCR text, then add the shapes fusion left out."""
        shape_by_box = {c.bbox.to_tuple(): c for c in cv_boxes}
        # Containers (present when fusion ran with include_containers=True) enclose their children's text, not their own
        nodes = [
            _Node(
                element=el,
                shape=shape_by_box.get(el.bbox.to_tuple()),
                fragments=[] if el.type == ElementType.CONTAINER else [
                    o for o in ocr_results if self._holds_text(el.bbox, o.bbox)
                ],
            )
            for el in state.elements
        ]

        # Containers give the structure; isolated shapes are controls whose text OCR missed (or visual noise)
        used = {n.shape.bbox.to_tuple() for n in nodes if n.shape is not None}
        leaf_boxes = [n.element.bbox for n in nodes if n.element.type != ElementType.CONTAINER]
        extras: list[_Node] = []
        for cv_box in cv_boxes:
            if cv_box.bbox.to_tuple() in used:
                continue
            if is_container_box(cv_box):
                elem_type = ElementType.CONTAINER
            elif not any(cv_box.bbox.intersects(b) for b in leaf_boxes):
                elem_type = ElementType.UNKNOWN
            else:
                continue
            extras.append(_Node(
                element=UIElement(id=-1, type=elem_type, bbox=cv_box.bbox, confidence=cv_box.confidence),
                shape=cv_box,
                fragments=[],
            ))

        extras.sort(key=lambda n: (n.element.bbox.y1 // 25, n.element.bbox.x1))
        next_id = max((n.element.id for n in nodes), default=-1) + 1
        for offset, node in enumerate(extras):
            node.element.id = next_id + offset
        return nodes + extras

    def _holds_text(self, box: BoundingBox, text_box: BoundingBox) -> bool:
        # Same rule fusion uses to attach OCR text to a detected box
        return box.contains(text_box, margin=self.containment_margin) or box.iou(text_box) > 0.4

    @staticmethod
    def _raw_detections(
        nodes: list[_Node],
        ocr_results: list[OCRResult],
        cv_boxes: list[DetectedBox],
    ) -> RawDetections:
        ocr_owner: dict[int, int] = {}
        shape_owner: dict[tuple[int, int, int, int], int] = {}
        for node in nodes:
            for fragment in node.fragments:
                ocr_owner.setdefault(id(fragment), node.element.id)
            if node.shape is not None:
                shape_owner[node.shape.bbox.to_tuple()] = node.element.id

        return RawDetections(
            ocr=[
                RawOCR(
                    text=o.text,
                    bbox=Box.of(o.bbox),
                    confidence=round(o.confidence, 3),
                    element_id=ocr_owner.get(id(o)),
                )
                for o in ocr_results
            ],
            cv=[
                RawShape(
                    bbox=Box.of(c.bbox),
                    category=c.category.value,
                    contour_depth=c.hierarchy_depth,
                    vertex_count=c.vertex_count,
                    element_id=shape_owner.get(c.bbox.to_tuple()),
                )
                for c in cv_boxes
            ],
        )


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #

def _assign_rows(ids: list[int], boxes: dict[int, BoundingBox]) -> dict[int, tuple[int, int]]:
    """Group sibling elements into visual rows, returning {id: (row, column)}."""
    rows: list[list[int]] = []
    for eid in sorted(ids, key=lambda i: (boxes[i].y1, boxes[i].x1)):
        box = boxes[eid]
        for row in rows:
            # Compare with the row's first element so a chain of overlaps can't merge distant rows
            anchor = boxes[row[0]]
            overlap = min(box.y2, anchor.y2) - max(box.y1, anchor.y1)
            if overlap >= 0.5 * max(1, min(box.height, anchor.height)):
                row.append(eid)
                break
        else:
            rows.append([eid])

    grid: dict[int, tuple[int, int]] = {}
    for r, row in enumerate(rows):
        for c, eid in enumerate(sorted(row, key=lambda i: boxes[i].x1)):
            grid[eid] = (r, c)
    return grid


def _ancestors(eid: int, parent_of: dict[int, Optional[int]]) -> list[int]:
    path: list[int] = []
    parent = parent_of[eid]
    while parent is not None:
        path.append(parent)
        parent = parent_of[parent]
    return path[::-1]


def _find_label(target: UIElement, texts: list[UIElement]) -> Optional[int]:
    """Nearest text to the left of or above the target, or to its right for checkboxes."""
    box = target.bbox
    best: Optional[tuple[float, int]] = None
    for text in texts:
        t = text.bbox
        y_overlap = min(box.y2, t.y2) - max(box.y1, t.y1)
        x_overlap = min(box.x2, t.x2) - max(box.x1, t.x1)
        scores = []
        if y_overlap > 5 and t.x2 <= box.x1 + 2 and box.x1 - t.x2 <= 150:
            scores.append(box.x1 - t.x2)
        if x_overlap > 5 and t.y2 <= box.y1 + 2 and box.y1 - t.y2 <= 50:
            scores.append((box.y1 - t.y2) * 1.5)
        if target.type == ElementType.CHECKBOX and y_overlap > 5 and t.x1 >= box.x2 - 2 and t.x1 - box.x2 <= 100:
            scores.append((t.x1 - box.x2) * 0.8)
        for score in scores:
            if best is None or score < best[0]:
                best = (score, text.id)
    return best[1] if best else None


def _neighbors(eid: int, pool: list[int], boxes: dict[int, BoundingBox]) -> Neighbors:
    box = boxes[eid]
    best: dict[str, tuple[int, int]] = {}

    def offer(direction: str, gap: int, other: int) -> None:
        if direction not in best or (gap, other) < best[direction]:
            best[direction] = (gap, other)

    for other in pool:
        if other == eid:
            continue
        o = boxes[other]
        if min(box.y2, o.y2) - max(box.y1, o.y1) > 0:
            if o.x2 <= box.x1 + 2:
                offer("left", box.x1 - o.x2, other)
            elif o.x1 >= box.x2 - 2:
                offer("right", o.x1 - box.x2, other)
        if min(box.x2, o.x2) - max(box.x1, o.x1) > 0:
            if o.y2 <= box.y1 + 2:
                offer("above", box.y1 - o.y2, other)
            elif o.y1 >= box.y2 - 2:
                offer("below", o.y1 - box.y2, other)

    return Neighbors(**{direction: other for direction, (_, other) in best.items()})


def _region(x: int, y: int, width: int, height: int) -> str:
    row = ("top", "middle", "bottom")[min(2, max(0, y * 3 // max(1, height)))]
    col = ("left", "center", "right")[min(2, max(0, x * 3 // max(1, width)))]
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


# --------------------------------------------------------------------------- #
# Shape and color helpers
# --------------------------------------------------------------------------- #

def _describe_shape(shape: Optional[DetectedBox], has_border: bool) -> Shape:
    if shape is None or shape.perimeter <= 0:
        return Shape(kind="text_region", has_border=has_border)

    # Contour points sit on pixel centers, so a solid w x h blob has polygon area (w-1)(h-1)
    box_area = max(1, (shape.bbox.width - 1) * (shape.bbox.height - 1))
    extent = min(1.0, shape.polygon_area / box_area)
    solidity = shape.polygon_area / shape.hull_area if shape.hull_area > 0 else 0.0
    circularity = min(1.0, 4 * math.pi * shape.polygon_area / shape.perimeter ** 2)
    return Shape(
        kind=_shape_kind(shape.vertex_count, extent, solidity, circularity),
        vertex_count=shape.vertex_count,
        extent=round(extent, 3),
        solidity=round(solidity, 3),
        circularity=round(circularity, 3),
        has_border=has_border,
    )


def _shape_kind(vertices: int, extent: float, solidity: float, circularity: float) -> ShapeKind:
    if 0.7 <= extent <= 0.83 and circularity >= 0.6:
        return "ellipse"
    if extent >= 0.85 and solidity >= 0.9:
        return "rectangle" if vertices == 4 and extent >= 0.95 else "rounded_rectangle"
    if solidity >= 0.9:
        return "polygon"
    return "irregular"


def _describe_style(
    rgb: np.ndarray,
    box: BoundingBox,
    fragments: list[OCRResult],
    outlined: bool,
) -> tuple[Style, bool]:
    """Colors of an element region; also returns whether it has a visible border."""
    crop = _crop(rgb, box)
    if crop.size == 0:
        return Style(background_color="#000000", brightness=0.0), False

    h, w = crop.shape[:2]
    inset = 3 if outlined and min(h, w) > 10 else 0
    interior = crop[inset:h - inset, inset:w - inset]
    background = _dominant_colors(interior.reshape(-1, 3))[0][0]

    border = None
    if outlined:
        # Edge detection widens boxes by about a pixel, so the drawn border sits in rings 1-2, not ring 0
        rings = [ring for ring in (_ring(crop, 1), _ring(crop, 2)) if len(ring)]
        candidates = [_dominant_colors(ring)[0][0] for ring in rings]
        if candidates:
            farthest = max(candidates, key=lambda c: _color_distance(c, background))
            if _color_distance(farthest, background) > 20:
                border = farthest

    text = _text_color(rgb, fragments, background)
    style = Style(
        background_color=_hex(background),
        border_color=_hex(border) if border is not None else None,
        text_color=_hex(text) if text is not None else None,
        brightness=round(float(crop.mean()) / 255.0, 3),
        contrast_ratio=_contrast_ratio(text, background) if text is not None else None,
    )
    return style, border is not None


def _describe_image(rgb: np.ndarray) -> ImageInfo:
    pixels = rgb.reshape(-1, 3)
    return ImageInfo(
        width=rgb.shape[1],
        height=rgb.shape[0],
        mean_color=_hex(pixels.mean(axis=0)),
        dominant_colors=[
            ColorShare(color=_hex(color), share=round(share, 4))
            for color, share in _dominant_colors(pixels, top=5)
        ],
    )


def _crop(rgb: np.ndarray, box: BoundingBox) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1 = max(0, box.x1), max(0, box.y1)
    x2, y2 = min(w, box.x2), min(h, box.y2)
    return rgb[y1:max(y1, y2), x1:max(x1, x2)]


def _ring(crop: np.ndarray, k: int) -> np.ndarray:
    """Pixels exactly k pixels in from the crop's edge, as an (N, 3) array."""
    h, w = crop.shape[:2]
    if h <= 2 * k + 1 or w <= 2 * k + 1:
        return np.empty((0, 3), dtype=crop.dtype)
    return np.concatenate([
        crop[k, k:w - k],
        crop[h - 1 - k, k:w - k],
        crop[k + 1:h - 1 - k, k],
        crop[k + 1:h - 1 - k, w - 1 - k],
    ])


def _dominant_colors(
    pixels: np.ndarray,
    top: int = 1,
    bin_size: int = 16,
    max_samples: int = 50_000,
) -> list[tuple[np.ndarray, float]]:
    """Most common colors as (mean RGB of the color bin, share of pixels), most common first."""
    if len(pixels) == 0:
        return [(np.zeros(3), 0.0)]
    if len(pixels) > max_samples:
        pixels = pixels[:: len(pixels) // max_samples + 1]

    bins = pixels.astype(np.int32) // bin_size
    keys = (bins[:, 0] << 16) | (bins[:, 1] << 8) | bins[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:top]
    return [(pixels[inverse == i].mean(axis=0), float(counts[i]) / len(keys)) for i in order]


def _text_color(rgb: np.ndarray, fragments: list[OCRResult], background: np.ndarray) -> Optional[np.ndarray]:
    if not fragments:
        return None
    pixels = np.concatenate([_crop(rgb, f.bbox).reshape(-1, 3) for f in fragments])
    distance = np.linalg.norm(pixels.astype(float) - background, axis=1)
    ink = distance > 40
    if ink.sum() < 8:
        return None
    # Anti-aliased glyph edges blend into the background; keep the most distinct half
    ink_distance = distance[ink]
    core = pixels[ink][ink_distance >= np.median(ink_distance)]
    return np.median(core, axis=0)


def _color_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _relative_luminance(color: np.ndarray) -> float:
    c = np.asarray(color, dtype=float) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _contrast_ratio(a: np.ndarray, b: np.ndarray) -> float:
    lighter, darker = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return round((lighter + 0.05) / (darker + 0.05), 2)


def _hex(color: np.ndarray) -> str:
    r, g, b = (min(255, max(0, int(round(float(c))))) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"
