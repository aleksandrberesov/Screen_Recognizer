"""Screen Recognizer: Fast, local, zero-LLM screenshot perception and UI element recognition engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from .annotator import VisualAnnotator
from .capture import ScreenCapture
from .detector import DetectedBox, UIElementDetector
from .fusion import SpatialFusionEngine
from .models import BoundingBox, ElementType, ScreenState, UIElement
from .ocr_engine import BaseOCREngine, OCRResult, create_ocr_engine
from .prompt_formatter import PromptFormatter
from .report import Point, ReportBuilder, ScreenReport, SourceInfo, ToolInfo

__version__ = "0.1.0"
__all__ = [
    "ScreenRecognizer",
    "RecognitionResult",
    "ScreenState",
    "UIElement",
    "BoundingBox",
    "ElementType",
    "VisualAnnotator",
    "PromptFormatter",
    "ScreenCapture",
    "ScreenReport",
    "ReportBuilder",
    "SourceInfo",
]


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


@dataclass
class RecognitionResult:
    """Output of one recognition pass: the fused state plus the raw detections behind it."""
    image: Image.Image
    state: ScreenState
    ocr_results: list[OCRResult]
    cv_boxes: list[DetectedBox]
    timings_ms: dict[str, float] = field(default_factory=dict)


class ScreenRecognizer:
    """
    Unified high-level interface for local screen perception.

    Coordinates screen capture, OCR, morphological CV detection, and spatial fusion.
    """

    def __init__(
        self,
        ocr_backend: str = "auto",
        max_edge: Optional[int] = None,
        enable_cv_detection: bool = True,
        include_containers: bool = False,
    ):
        """
        :param ocr_backend: 'auto', 'rapidocr', or 'tesseract'
        :param max_edge: Downscale limit for OCR speedups on ultra-high-res screens (None = native)
        :param enable_cv_detection: Whether to run OpenCV contour & icon detector alongside OCR
        :param include_containers: Whether to output enclosing container rectangles in ScreenState
        """
        self.ocr_engine: BaseOCREngine = create_ocr_engine(backend=ocr_backend, max_edge=max_edge)
        self.enable_cv_detection = enable_cv_detection
        self.detector = UIElementDetector() if enable_cv_detection else None
        self.fusion = SpatialFusionEngine(include_containers=include_containers)
        self.annotator = VisualAnnotator()

    def recognize(
        self,
        source: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        bbox: Optional[Union[BoundingBox, Sequence[int]]] = None,
        all_screens: bool = True,
    ) -> ScreenState:
        """
        Recognize UI elements from a live screen, bounding box, or provided image.

        :param source: None (captures current screen), or filepath / PIL Image / numpy array.
        :param bbox: Optional region bounding box (left, top, right, bottom) if capturing screen.
        :param all_screens: If capturing full desktop across multi-monitor setups.
        :return: Parsed ScreenState.
        """
        if source is None:
            image = ScreenCapture.capture_screen(bbox=bbox, all_screens=all_screens)
        else:
            image = ScreenCapture.load_image(source)

        return self.analyze(image).state

    def analyze(self, image: Union[Image.Image, np.ndarray]) -> RecognitionResult:
        """
        Run OCR, CV detection and fusion on an image, keeping raw detections and per-stage timings.
        """
        image = ScreenCapture.load_image(image)
        w, h = image.size
        timings: dict[str, float] = {}

        # 1. Run Local OCR
        start = time.perf_counter()
        ocr_results = self.ocr_engine.recognize(image)
        timings["ocr"] = _elapsed_ms(start)

        # 2. Run Local CV Detector (if enabled)
        start = time.perf_counter()
        cv_boxes: list[DetectedBox] = self.detector.detect(image) if self.detector is not None else []
        timings["detection"] = _elapsed_ms(start)

        # 3. Spatial Fusion
        start = time.perf_counter()
        state = self.fusion.fuse(
            ocr_results=ocr_results,
            cv_boxes=cv_boxes,
            screen_size=(w, h),
        )
        timings["fusion"] = _elapsed_ms(start)

        return RecognitionResult(
            image=image,
            state=state,
            ocr_results=ocr_results,
            cv_boxes=cv_boxes,
            timings_ms=timings,
        )

    def report(
        self,
        source: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        bbox: Optional[Union[BoundingBox, Sequence[int]]] = None,
        all_screens: bool = True,
        window_title: Optional[str] = None,
        hwnd: Optional[int] = None,
        include_raw: bool = True,
    ) -> ScreenReport:
        """
        Build a full report: every element with its hierarchy, geometry, shape, colors and relations.

        :param source: None (captures current screen), or filepath / PIL Image / numpy array.
        :param bbox: Optional region bounding box (left, top, right, bottom) if capturing screen.
        :param all_screens: If capturing full desktop across multi-monitor setups.
        :param window_title: Capture the window whose title contains this text instead of the screen.
        :param hwnd: Capture the window with this handle instead of the screen.
        :param include_raw: Include the unfused OCR and CV detections.
        """
        start = time.perf_counter()
        if window_title is not None or hwnd is not None:
            image, window_bbox = ScreenCapture.capture_window(window_title=window_title, hwnd=hwnd)
            source_info = SourceInfo(
                kind="window",
                window_title=window_title,
                offset=Point(x=window_bbox.x1, y=window_bbox.y1),
            )
        elif source is None:
            image = ScreenCapture.capture_screen(bbox=bbox, all_screens=all_screens)
            x, y = ScreenCapture.screen_origin(bbox=bbox, all_screens=all_screens)
            source_info = SourceInfo(kind="screen", offset=Point(x=x, y=y))
        else:
            image = ScreenCapture.load_image(source)
            is_file = isinstance(source, (str, Path))
            source_info = SourceInfo(kind="file" if is_file else "image", path=str(source) if is_file else None)
        capture_ms = _elapsed_ms(start)

        recognition = self.analyze(image)
        recognition.timings_ms["capture"] = capture_ms
        return self.build_report(recognition, source=source_info, include_raw=include_raw)

    def build_report(
        self,
        recognition: RecognitionResult,
        source: Optional[SourceInfo] = None,
        include_raw: bool = True,
    ) -> ScreenReport:
        """
        Build a full report from an analyze() result, without running recognition again.
        """
        return ReportBuilder().build(
            image=recognition.image,
            state=recognition.state,
            ocr_results=recognition.ocr_results,
            cv_boxes=recognition.cv_boxes,
            source=source or SourceInfo(kind="image"),
            tool=ToolInfo(
                version=__version__,
                ocr_backend=self.ocr_engine.name,
                cv_detection=self.detector is not None,
            ),
            timings_ms=recognition.timings_ms,
            include_raw=include_raw,
        )

    def recognize_window(
        self,
        window_title: Optional[str] = None,
        hwnd: Optional[int] = None,
    ) -> ScreenState:
        """
        Capture and recognize UI elements inside a specific application window.
        """
        image, window_bbox = ScreenCapture.capture_window(window_title=window_title, hwnd=hwnd)
        state = self.recognize(source=image)
        # Store window screen offset in metadata so clickers know absolute screen coordinates
        state.metadata["window_bbox"] = window_bbox.to_dict()
        state.metadata["offset_x"] = window_bbox.x1
        state.metadata["offset_y"] = window_bbox.y1
        return state

    def annotate(
        self,
        state: ScreenState,
        image: Optional[Image.Image] = None,
        output_path: Optional[str] = None,
        only_interactive: bool = False,
    ) -> Image.Image:
        """
        Produce a Set-of-Marks annotated image overlay with numbered badges.
        """
        if image is None:
            image = ScreenCapture.capture_screen()
        return self.annotator.annotate(
            image=image,
            state=state,
            output_path=output_path,
            only_interactive=only_interactive,
        )

    def to_prompt(
        self,
        state: ScreenState,
        goal: Optional[str] = None,
        only_interactive: bool = True,
    ) -> str:
        """
        Format the screen state into a concise LLM decision-engine prompt.
        """
        return PromptFormatter.format_for_llm(state, goal=goal, only_interactive=only_interactive)
