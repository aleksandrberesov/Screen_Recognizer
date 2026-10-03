# Screen Recognizer

**Fast, local, zero-LLM screenshot perception and UI element recognition engine.**

Screen Recognizer is an independent, offline Python library and CLI designed to bridge the gap between pixel-based user interfaces and autonomous agents (or automation tools like App Clickers and RPA bots).

It performs **100% local visual perception** using local ONNX-based OCR (`RapidOCR`) and OpenCV computer vision:
- Detects buttons, input fields, checkboxes, icons, and text labels.
- Computes exact pixel coordinates `(x, y)` and bounding boxes.
- Formats structured state outputs (`JSON`, `dict`, and token-optimized `Markdown` tables for LLM prompts).
- Renders Set-of-Marks (SoM) visual overlays with high-contrast numbered badges.
- Produces a full JSON report (hierarchy, shapes, colors, layout, relations) with a JSON Schema, for handing to other tools.
- Eliminates LLM coordinate hallucination by allowing the LLM to function **strictly as a decision engine** (the LLM only picks an element ID or action, while this tool maps it to exact pixels).

---

## Architecture

```
[ Screenshot / Window / Image File ]
                 │
                 ├──► Local OCR (RapidOCR / ONNX) ──► Text & Bounding Boxes
                 └──► Local CV (OpenCV Contours)  ──► UI Boxes, Inputs & Icons
                                  │
                  Spatial Fusion & Deduplication
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
   [ Structured ScreenState ]             [ Set-of-Marks Image ]
   • elements: list[UIElement]            (Numbered badges for debugging
   • .to_json() / .to_dict()               or multimodal inspection)
   • .to_markdown() for LLM prompts
   • .find_by_text(), .find_by_id()
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
[ LLM Decision Engine ]   [ App Clicker / RPA ]
(Chooses element ID)      (Clicks exact center X, Y)
```

---

## Installation

Inside your project or shared virtual environment:

```bash
cd E:\Automation_projects\Screen_Recognizer
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

### Dependencies
- `pillow>=10.0`
- `numpy>=1.24`
- `opencv-python>=4.8`
- `pydantic>=2.0`
- `rapidocr-onnxruntime==1.2.3` (Pure pip, runs on CPU/GPU without external binaries)

---

## Quick Start (Python API)

### 1. Basic Recognition
```python
from screen_recognizer import ScreenRecognizer

recognizer = ScreenRecognizer()

# Recognize from live screen, window, or image file
state = recognizer.recognize(source="path/to/screenshot.png")  # Or source=None for live desktop

print(f"Screen: {state.width}x{state.height}")
print(f"Found {len(state.elements)} elements:")

for el in state.get_clickable():
    print(f"[{el.id}] {el.type.value:<8} '{el.text}' at ({el.center_x}, {el.center_y})")
```

### 2. Finding and Locating Elements
```python
# Exact or fuzzy search by text
buttons = state.find_by_text("Sign In", fuzzy=True)
if buttons:
    target = buttons[0]
    print(f"Click target at: ({target.center_x}, {target.center_y})")

# Find by ID
element = state.find_by_id(5)

# Find closest element to a coordinate
nearby = state.find_closest(x=100, y=200, elem_type="button")
```

### 3. Set-of-Marks Visual Grounding
```python
# Generates an annotated screenshot with numbered badges [0], [1], [2]...
annotated_img = recognizer.annotate(
    state=state,
    output_path="annotated_screen.png",
    only_interactive=True  # Focus only on buttons, inputs, checkboxes
)
```

### 4. Feeding into an LLM Decision Engine
```python
# Generate a compact, token-optimized table for LLM prompts
prompt = recognizer.to_prompt(state, goal="Log in using credentials")
print(prompt)

# Example prompt output:
# Screen Resolution: 1000x700
# Available Interactive Elements (11 items):
# - [3] INPUT "Search projects.." | Center: (765, 60)
# - [10] BUTTON "Run Test Suite" | Center: (610, 195)
# - [12] INPUT "alex@automation.io" | Center: (240, 213)
# - [14] INPUT "password123" | Center: (240, 293)
# - [15] BUTTON "Sign In" | Center: (120, 348)
#
# Current Goal: Log in using credentials
# Respond with the next action in JSON format: {"action": "click", "element_id": 15}
```

### 5. Full Report for Other Tools
`report()` puts everything the pipeline knows about the screen into one JSON document with a published schema. It includes every element `recognize()` finds, plus containers (cards, panels, menu bars) and outlined shapes whose text OCR missed.

```python
report = recognizer.report(source="path/to/screenshot.png")  # or window_title="Calculator", or no source for the live screen
report.save("screen_report.json")

button = report.find_by_id(15)
print(button.screen_center, button.hierarchy.parent_id, button.style.background_color)
```

If you already called `recognizer.analyze(image)`, build the report from that result without running OCR again: `recognizer.build_report(recognition)`. To reuse `annotate()` or `find_by_text()` on every report element (same IDs, containers included), call `report.to_screen_state()`.

Each element contains:

| Field | Contents |
|---|---|
| `id`, `type`, `text`, `interactive`, `confidence` | As in `recognize()`, with the same IDs. Containers and shapes `recognize()` doesn't return are numbered after them |
| `bbox`, `center`, `screen_center` | Image pixels. `screen_center` adds the window or monitor offset; use it to click |
| `geometry` | Width, height, area, aspect ratio, screen coverage, `touches_edge` |
| `shape` | `kind` (`rectangle`, `rounded_rectangle`, `ellipse`, `polygon`, `irregular`, `text_region`), vertex count, extent, solidity, circularity, `has_border` |
| `style` | Background, border and text colors (`#rrggbb`), brightness, WCAG contrast ratio |
| `text_fragments` | Raw OCR lines merged into the element |
| `hierarchy` | `parent_id`, `children_ids`, `depth`, `path` |
| `layout` | `row` and `column` among siblings, screen `region` |
| `relations` | Nearest `neighbors` left/right/above/below, `labelled_by`, `label_for` |
| `provenance` | Which detectors found it (`ocr`, `cv`) and the raw CV category |
| `attributes` | Extra facts, such as a container's `title` |

The report also has `source` (kind, path, window title, screen offset), `image` (size, dominant colors), `timings_ms`, `summary` (counts by type, max depth), `tree` (nested structure) and `raw` (unfused OCR and CV detections; pass `include_raw=False` to leave them out).

For validation or code generation, get the JSON Schema from `ScreenReport.model_json_schema()` or `python -m screen_recognizer.cli schema -o report.schema.json`.

---

## Downstream Integration: App Clicker

Example of how `App_Clicker` or any automation tool uses `ScreenRecognizer` to click text:

```python
from screen_recognizer import ScreenRecognizer
import pyautogui  # or Windows ctypes

recognizer = ScreenRecognizer()
state = recognizer.recognize()

matches = state.find_by_text("Sign In")
if matches:
    pyautogui.click(matches[0].center_x, matches[0].center_y)
```

---

## Command Line Interface (CLI)

Screen Recognizer includes a CLI for quick manual testing and automation pipelines:

### Capture Live Desktop
```bash
python -m screen_recognizer.cli capture --markdown --interactive-only --output screen.json --annotate screen_marked.png
```

### Analyze an Image File
```bash
python -m screen_recognizer.cli file path/to/app.png --markdown --annotate output_marked.png
```

### Capture Specific Window
```bash
python -m screen_recognizer.cli window "Calculator" --output calc_state.json
```

### Full Report
Add `--report` to `capture`, `file` or `window`:
```bash
python -m screen_recognizer.cli window "Calculator" --report calc_report.json
python -m screen_recognizer.cli schema --output report.schema.json
```

---

## Examples & Tests

The `examples/` directory contains runnable scripts:
- `examples/01_basic_usage.py`: Basic capture and text query.
- `examples/02_llm_decision_engine.py`: How to pass parsed state to an LLM planner and parse actions.
- `examples/03_app_clicker_integration.py`: How a clicker consumes coordinates.

Run tests:
```bash
python tests/test_recognition.py
python tests/generate_and_test_ui.py
python tests/test_report.py
```
