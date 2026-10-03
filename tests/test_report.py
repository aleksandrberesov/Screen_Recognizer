"""Tests for the full screen report."""

import json
import sys
from pathlib import Path

# Add project root and tests directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw

from generate_and_test_ui import generate_mock_app_image
from screen_recognizer import ScreenRecognizer, ScreenReport, SourceInfo
from screen_recognizer.capture import ScreenCapture
from screen_recognizer.fusion import SpatialFusionEngine
from screen_recognizer.models import ElementType
from screen_recognizer.report import Point, ReportBuilder

_cache = {}


def _recognizer() -> ScreenRecognizer:
    if "recognizer" not in _cache:
        _cache["recognizer"] = ScreenRecognizer()
    return _cache["recognizer"]


def _sample():
    """(recognition, report) for the mock dashboard, computed once."""
    if "sample" not in _cache:
        recognition = _recognizer().analyze(generate_mock_app_image())
        _cache["sample"] = (recognition, _recognizer().build_report(recognition))
    return _cache["sample"]


def _by_text(report: ScreenReport, text: str):
    return next(el for el in report.elements if text.lower() in el.text.lower())


def _close(hex_color: str, rgb: tuple, tolerance: float = 24) -> bool:
    actual = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return sum((a - b) ** 2 for a, b in zip(actual, rgb)) ** 0.5 <= tolerance


def test_ids_match_recognize():
    recognition, report = _sample()
    for el in recognition.state.elements:
        match = report.find_by_id(el.id)
        assert match is not None, f"Element {el.id} missing from report"
        assert (match.type, match.text) == (el.type, el.text)


def test_hierarchy():
    _, report = _sample()
    containers = [el for el in report.elements if el.type == ElementType.CONTAINER]
    assert len(containers) >= 3, "Menu bar and both cards should be containers"

    sign_in = _by_text(report, "Sign In")
    cancel = _by_text(report, "Cancel")
    card = report.find_by_id(sign_in.hierarchy.parent_id)
    assert card is not None and card.type == ElementType.CONTAINER
    assert "Auth" in card.attributes.get("title", "")
    assert cancel.hierarchy.parent_id == card.id
    assert {sign_in.id, cancel.id} <= set(card.hierarchy.children_ids)
    assert sign_in.hierarchy.path == [card.id]

    # The tree holds every element exactly once
    seen = []

    def walk(nodes):
        for node in nodes:
            seen.append(node.id)
            walk(node.children)

    walk(report.tree)
    assert sorted(seen) == [el.id for el in report.elements]


def test_layout_and_relations():
    _, report = _sample()
    sign_in = _by_text(report, "Sign In")
    cancel = _by_text(report, "Cancel")

    assert sign_in.layout.row == cancel.layout.row
    assert (sign_in.layout.column, cancel.layout.column) == (0, 1)
    assert sign_in.relations.neighbors.right == cancel.id
    assert cancel.relations.neighbors.left == sign_in.id

    username_input = _by_text(report, "alex@")
    label = report.find_by_id(username_input.relations.labelled_by)
    assert label is not None and "Username" in label.text
    assert username_input.id in label.relations.label_for


def test_shape_and_style():
    _, report = _sample()
    # Blue Sign In button: fill (37, 99, 235), outline (29, 78, 216), white text
    sign_in = _by_text(report, "Sign In")
    assert sign_in.shape.kind == "rectangle", sign_in.shape
    assert sign_in.shape.has_border
    assert _close(sign_in.style.background_color, (37, 99, 235)), sign_in.style
    assert _close(sign_in.style.text_color, (255, 255, 255), tolerance=60), sign_in.style
    assert sign_in.style.contrast_ratio > 3

    # Plain text has no outline analysis, including flat text buttons whose only outline is their glyphs
    assert _by_text(report, "Password:").shape.kind == "text_region"
    assert _by_text(report, "Help").shape.kind == "text_region"

    # The menu bar spans the full width, so its outline is cut off at the image edge
    menu_bar = report.find_by_id(_by_text(report, "File").hierarchy.parent_id)
    assert menu_bar.geometry.touches_edge
    assert not sign_in.geometry.touches_edge


def test_shape_kinds():
    img = Image.new("RGB", (600, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 200, 90], outline=(40, 40, 40), width=2)
    draw.rounded_rectangle([260, 40, 420, 90], radius=16, outline=(40, 40, 40), width=2)
    draw.ellipse([480, 30, 560, 110], outline=(40, 40, 40), width=2)

    report = _recognizer().report(source=img)

    def kind_at(x, y):
        el = min(report.elements, key=lambda e: (e.center.x - x) ** 2 + (e.center.y - y) ** 2)
        return el.shape.kind

    assert kind_at(120, 65) == "rectangle"
    assert kind_at(340, 65) == "rounded_rectangle"
    assert kind_at(520, 70) == "ellipse"


def test_serialization():
    _, report = _sample()
    data = json.loads(report.to_json())
    assert data["report_format"] == "screen-recognizer.report"
    assert data["elements"][0]["type"] in {t.value for t in ElementType}
    assert ScreenReport.model_validate_json(report.to_json()) == report

    schema = ScreenReport.model_json_schema()
    assert {"elements", "tree", "summary"} <= set(schema["properties"])


def test_to_screen_state():
    recognition, report = _sample()
    state = report.to_screen_state()
    assert [el.id for el in state.elements] == [el.id for el in report.elements]
    assert state.find_by_text("Sign In")
    assert state.find_by_type(ElementType.CONTAINER)
    annotated = _recognizer().annotate(state, image=recognition.image)
    assert annotated.size == recognition.image.size


def test_report_unchanged_by_include_containers():
    recognition, report = _sample()
    nested_state = SpatialFusionEngine(include_containers=True).fuse(
        recognition.ocr_results, recognition.cv_boxes, recognition.image.size
    )
    nested = ReportBuilder().build(
        image=recognition.image,
        state=nested_state,
        ocr_results=recognition.ocr_results,
        cv_boxes=recognition.cv_boxes,
        source=report.source,
        tool=report.tool,
    )

    def signature(r: ScreenReport):
        # IDs differ between the two states, so identify elements and their parents by bounding box
        bbox_of = {el.id: (el.bbox.x1, el.bbox.y1, el.bbox.x2, el.bbox.y2) for el in r.elements}
        return sorted(
            (
                el.type.value, el.text, bbox_of[el.id], bbox_of.get(el.hierarchy.parent_id, ()),
                el.shape.model_dump_json(), el.style.model_dump_json(), el.layout.model_dump_json(),
                [f.text for f in el.text_fragments], el.attributes.get("title"),
            )
            for el in r.elements
        )

    assert signature(nested) == signature(report)


def test_screen_center_uses_source_offset():
    recognition, _ = _sample()
    source = SourceInfo(kind="window", window_title="Demo", offset=Point(x=100, y=200))
    report = _recognizer().build_report(recognition, source=source, include_raw=False)
    assert report.raw is None
    for el in report.elements:
        assert (el.screen_center.x, el.screen_center.y) == (el.center.x + 100, el.center.y + 200)


def test_screen_origin_for_region():
    assert ScreenCapture.screen_origin(bbox=(100, 50, 400, 300)) == (100, 50)
    assert ScreenCapture.screen_origin(all_screens=False) == (0, 0)


if __name__ == "__main__":
    print("Running report tests...")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}")
    print("\nAll report tests passed successfully!")
