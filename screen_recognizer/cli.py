"""Command-line interface for local screen recognition."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from . import ScreenRecognizer
from .capture import ScreenCapture
from .report import Point, ScreenReport, SourceInfo


def main():
    parser = argparse.ArgumentParser(
        description="Local Screenshot Recognition Tool (Zero-LLM GUI Perception Engine)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: capture
    cap_parser = subparsers.add_parser("capture", help="Capture current screen and recognize elements")
    cap_parser.add_argument("--output", "-o", type=str, help="Save structured state to JSON file")
    cap_parser.add_argument("--report", "-r", type=str, help="Save full report (hierarchy, shapes, colors) to JSON file")
    cap_parser.add_argument("--annotate", "-a", type=str, help="Save annotated Set-of-Mark image (PNG/JPG)")
    cap_parser.add_argument("--markdown", "-m", action="store_true", help="Print LLM Markdown prompt to stdout")
    cap_parser.add_argument("--ocr-backend", choices=["auto", "rapidocr", "tesseract"], default="auto")
    cap_parser.add_argument("--interactive-only", action="store_true", help="Only show clickable/input elements")

    # Command: file
    file_parser = subparsers.add_parser("file", help="Recognize elements from an existing image file")
    file_parser.add_argument("image_path", type=str, help="Path to input screenshot image")
    file_parser.add_argument("--output", "-o", type=str, help="Save structured state to JSON file")
    file_parser.add_argument("--report", "-r", type=str, help="Save full report (hierarchy, shapes, colors) to JSON file")
    file_parser.add_argument("--annotate", "-a", type=str, help="Save annotated Set-of-Mark image")
    file_parser.add_argument("--markdown", "-m", action="store_true", help="Print LLM Markdown prompt to stdout")
    file_parser.add_argument("--ocr-backend", choices=["auto", "rapidocr", "tesseract"], default="auto")
    file_parser.add_argument("--interactive-only", action="store_true", help="Only show clickable/input elements")

    # Command: window
    win_parser = subparsers.add_parser("window", help="Capture specific window by title")
    win_parser.add_argument("title", type=str, help="Substring of window title to capture")
    win_parser.add_argument("--output", "-o", type=str, help="Save structured state to JSON file")
    win_parser.add_argument("--report", "-r", type=str, help="Save full report (hierarchy, shapes, colors) to JSON file")
    win_parser.add_argument("--annotate", "-a", type=str, help="Save annotated Set-of-Mark image")
    win_parser.add_argument("--markdown", "-m", action="store_true", help="Print LLM Markdown prompt to stdout")
    win_parser.add_argument("--ocr-backend", choices=["auto", "rapidocr", "tesseract"], default="auto")

    # Command: schema
    schema_parser = subparsers.add_parser("schema", help="Print the JSON Schema of the full report")
    schema_parser.add_argument("--output", "-o", type=str, help="Save the schema to a file instead of printing it")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "schema":
        schema = json.dumps(ScreenReport.model_json_schema(), indent=2)
        if args.output:
            Path(args.output).write_text(schema, encoding="utf-8")
            print(f"Report schema saved to {args.output}")
        else:
            print(schema)
        return

    recognizer = ScreenRecognizer(ocr_backend=args.ocr_backend)

    t0 = time.perf_counter()

    if args.command == "capture":
        print("Capturing live desktop screen...")
        raw_img = ScreenCapture.capture_screen()
        x, y = ScreenCapture.screen_origin()
        source = SourceInfo(kind="screen", offset=Point(x=x, y=y))
    elif args.command == "file":
        img_path = Path(args.image_path)
        if not img_path.exists():
            print(f"Error: file not found '{img_path}'", file=sys.stderr)
            sys.exit(1)
        raw_img = Image.open(img_path)
        source = SourceInfo(kind="file", path=str(img_path))
    elif args.command == "window":
        print(f"Capturing window matching '{args.title}'...")
        raw_img, win_bbox = ScreenCapture.capture_window(window_title=args.title)
        source = SourceInfo(kind="window", window_title=args.title, offset=Point(x=win_bbox.x1, y=win_bbox.y1))

    capture_ms = (time.perf_counter() - t0) * 1000.0
    recognition = recognizer.analyze(raw_img)
    state = recognition.state

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Recognition completed in {elapsed_ms:.1f}ms")
    print(f"Found {len(state.elements)} elements ({len(state.get_clickable())} interactive)")

    if getattr(args, "markdown", False):
        print("\n--- LLM PROMPT VIEW ---")
        print(state.to_markdown(only_interactive=getattr(args, "interactive_only", False)))
        print("------------------------\n")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(state.to_json(), encoding="utf-8")
        print(f"State saved to {out_path}")

    if args.report:
        recognition.timings_ms["capture"] = capture_ms
        report = recognizer.build_report(recognition, source=source)
        report_path = report.save(args.report)
        print(f"Full report saved to {report_path} ({report.summary.total_elements} elements, "
              f"depth {report.summary.max_depth})")

    if args.annotate:
        ann_path = Path(args.annotate)
        recognizer.annotator.annotate(
            image=raw_img,
            state=state,
            output_path=str(ann_path),
            only_interactive=getattr(args, "interactive_only", False),
        )
        print(f"Annotated Set-of-Marks image saved to {ann_path}")


if __name__ == "__main__":
    main()
