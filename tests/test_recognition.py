"""Unit and integration tests for screen_recognizer."""

import sys
from pathlib import Path

# Add project root and tests directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont

from generate_and_test_ui import generate_mock_app_image
from screen_recognizer.fusion import SpatialFusionEngine
from screen_recognizer.models import BoundingBox, ElementType, ScreenState, UIElement
from screen_recognizer.prompt_formatter import PromptFormatter
from screen_recognizer import ScreenRecognizer


def test_bounding_box_geometry():
    box1 = BoundingBox(10, 10, 50, 50)
    box2 = BoundingBox(20, 20, 40, 40)
    box3 = BoundingBox(100, 100, 150, 150)

    assert box1.width == 40
    assert box1.height == 40
    assert box1.area == 1600
    assert box1.center == (30, 30)

    # Containment
    assert box1.contains(box2) is True
    assert box2.contains(box1) is False
    assert box1.contains(box3) is False

    # Intersection
    assert box1.intersects(box2) is True
    assert box1.intersects(box3) is False

    # IoU
    iou = box1.iou(box2)
    assert 0.0 < iou <= 1.0


def test_screen_state_queries():
    elements = [
        UIElement(id=0, type=ElementType.TEXT, text="Sign in to your account", bbox=BoundingBox(10, 10, 200, 30)),
        UIElement(id=1, type=ElementType.INPUT, text="Username", bbox=BoundingBox(10, 50, 200, 80)),
        UIElement(id=2, type=ElementType.BUTTON, text="Login Now", bbox=BoundingBox(10, 100, 120, 140)),
        UIElement(id=3, type=ElementType.ICON, text="Help", bbox=BoundingBox(300, 10, 330, 40)),
    ]
    state = ScreenState(elements=elements, screen_size=(800, 600))

    # Test find_by_id
    assert state.find_by_id(2).text == "Login Now"
    assert state.find_by_id(99) is None

    # Test exact and fuzzy find_by_text
    exact = state.find_by_text("Login Now", exact=True)
    assert len(exact) == 1
    assert exact[0].id == 2

    fuzzy = state.find_by_text("Login")
    assert len(fuzzy) >= 1
    assert fuzzy[0].id == 2

    # Test get_clickable
    clickable = state.get_clickable()
    assert len(clickable) == 3  # INPUT, BUTTON, ICON

    # Test find_closest
    closest = state.find_closest(12, 102)
    assert closest.id == 2

    # Test serialization
    md = state.to_markdown()
    assert "[2]" in md
    assert "Login Now" in md

    js = state.to_json()
    assert '"elements_count": 4' in js


def test_prompt_formatter():
    elements = [
        UIElement(id=0, type=ElementType.BUTTON, text="Submit", bbox=BoundingBox(50, 50, 150, 90)),
    ]
    state = ScreenState(elements=elements, screen_size=(500, 500))

    prompt = PromptFormatter.format_for_llm(state, goal="Click submit")
    assert "[0] BUTTON \"Submit\"" in prompt
    assert "Click submit" in prompt

    # Test action parsing
    response = '{"action": "click", "element_id": 0}'
    action, target = PromptFormatter.parse_action_response(response, state)
    assert action["action"] == "click"
    assert target is not None
    assert target.id == 0
    assert target.center == (100, 70)


def test_synthetic_image_recognition():
    """Create a synthetic image with text and button boxes to verify end-to-end OCR and detection."""
    img = Image.new("RGB", (600, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw header text
    draw.text((50, 30), "Screen Recognizer Test Suite", fill=(20, 20, 20))

    # Draw a distinct button rectangle with border
    draw.rectangle([50, 100, 200, 150], fill=(70, 130, 180), outline=(30, 60, 100), width=2)
    draw.text((85, 115), "OK CONFIRM", fill=(255, 255, 255))

    # Draw another button
    draw.rectangle([250, 100, 400, 150], fill=(220, 80, 80), outline=(150, 40, 40), width=2)
    draw.text((295, 115), "CANCEL", fill=(255, 255, 255))

    recognizer = ScreenRecognizer()
    state = recognizer.recognize(source=img)

    assert len(state.elements) > 0

    # Search for text
    ok_matches = state.find_by_text("CONFIRM")
    assert len(ok_matches) > 0, "Failed to find 'CONFIRM' button via OCR"

    cancel_matches = state.find_by_text("CANCEL")
    assert len(cancel_matches) > 0, "Failed to find 'CANCEL' button via OCR"

    # Test annotation on synthetic image
    annotated = recognizer.annotate(state, image=img)
    assert annotated.size == (600, 400)
    print("\n[SUCCESS] Synthetic image recognition test passed!")


def test_include_containers_keeps_children():
    """Containers must be added alongside the elements inside them, not replace them."""
    img = generate_mock_app_image()
    recognizer = ScreenRecognizer()
    ocr_results = recognizer.ocr_engine.recognize(img)
    cv_boxes = recognizer.detector.detect(img)

    flat = SpatialFusionEngine(include_containers=False).fuse(ocr_results, cv_boxes, img.size)
    nested = SpatialFusionEngine(include_containers=True).fuse(ocr_results, cv_boxes, img.size)

    def controls(state):
        return sorted(
            (el.type.value, el.text, el.bbox.to_tuple())
            for el in state.elements
            if el.type != ElementType.CONTAINER
        )

    assert controls(nested) == controls(flat)
    for text in ("File", "Cancel", "Password:", "Quick Actions"):
        assert nested.find_by_text(text), f"'{text}' lost when containers are included"

    containers = nested.find_by_type(ElementType.CONTAINER)
    assert len(containers) >= 3, "Menu bar and both cards should be containers"
    assert all(el.text == "" for el in containers)
    assert not flat.find_by_type(ElementType.CONTAINER)
    assert [el.id for el in nested.elements] == list(range(len(nested.elements)))


if __name__ == "__main__":
    print("Running tests...")
    test_bounding_box_geometry()
    print("[OK] BoundingBox tests passed")
    test_screen_state_queries()
    print("[OK] ScreenState queries passed")
    test_prompt_formatter()
    print("[OK] PromptFormatter tests passed")
    test_synthetic_image_recognition()
    print("[OK] Synthetic image OCR & recognition passed")
    test_include_containers_keeps_children()
    print("[OK] include_containers keeps child elements")
    print("\nAll tests passed successfully!")
