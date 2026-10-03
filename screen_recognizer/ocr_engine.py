"""Local OCR engine with RapidOCR (ONNX) and Tesseract backends."""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .models import BoundingBox


@dataclass
class OCRResult:
    """Individual text block found by OCR."""
    text: str
    bbox: BoundingBox
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center


class BaseOCREngine(ABC):
    name = "base"

    @abstractmethod
    def recognize(self, image: Image.Image) -> list[OCRResult]:
        """Extract text blocks with bounding boxes and confidence scores."""
        pass


class RapidOCREngine(BaseOCREngine):
    """
    High-accuracy deep-learning OCR powered by ONNX Runtime.
    Requires no external binaries, runs completely local.
    """

    name = "rapidocr"

    def __init__(self, max_edge: Optional[int] = None):
        """
        :param max_edge: Optional maximum edge length to downscale before running OCR,
                         for speedups on 4K/retina displays. Coordinates are scaled back
                         to original resolution. If None, runs at native resolution.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "rapidocr-onnxruntime is not installed. Install via:\n"
                "pip install rapidocr-onnxruntime==1.2.3"
            )
        self._engine = RapidOCR()
        self.max_edge = max_edge

    def recognize(self, image: Image.Image) -> list[OCRResult]:
        img_np = np.array(image.convert("RGB"))
        # Convert RGB to BGR for OpenCV / RapidOCR
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = bgr.shape[:2]

        scale = 1.0
        if self.max_edge and max(orig_h, orig_w) > self.max_edge:
            scale = self.max_edge / max(orig_h, orig_w)
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        results, _ = self._engine(bgr)
        if not results:
            return []

        out: list[OCRResult] = []
        for box_pts, text, score in results:
            txt = (text or "").strip()
            if not txt:
                continue

            # Scale coordinates back to original resolution
            xs = [p[0] / scale for p in box_pts]
            ys = [p[1] / scale for p in box_pts]

            bbox = BoundingBox(
                int(round(min(xs))),
                int(round(min(ys))),
                int(round(max(xs))),
                int(round(max(ys))),
            )
            out.append(OCRResult(text=txt, bbox=bbox, confidence=float(score)))

        return out


class TesseractOCREngine(BaseOCREngine):
    """
    Classical Tesseract OCR backend via pytesseract.
    Requires the Tesseract binary to be installed on the system.
    """

    name = "tesseract"

    def __init__(self, tesseract_cmd: Optional[str] = None):
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract is not installed. Install via: pip install pytesseract")

        cmd = tesseract_cmd or self._find_tesseract_binary()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self._pt = pytesseract

    @staticmethod
    def _find_tesseract_binary() -> Optional[str]:
        exe = shutil.which("tesseract")
        if exe:
            return exe
        for path in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ):
            if os.path.isfile(path):
                return path
        return None

    def recognize(self, image: Image.Image) -> list[OCRResult]:
        from pytesseract import Output

        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        data = self._pt.image_to_data(gray, output_type=Output.DICT)

        # Group words on the same line into coherent phrases
        lines: dict[tuple[int, int, int], dict] = {}
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            txt = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i]) / 100.0
            except (ValueError, TypeError):
                conf = -1.0

            if not txt or conf <= 0.3:
                continue

            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]

            if key not in lines:
                lines[key] = {
                    "words": [txt],
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "confs": [conf],
                }
            else:
                entry = lines[key]
                entry["words"].append(txt)
                entry["x1"] = min(entry["x1"], x)
                entry["y1"] = min(entry["y1"], y)
                entry["x2"] = max(entry["x2"], x + w)
                entry["y2"] = max(entry["y2"], y + h)
                entry["confs"].append(conf)

        out: list[OCRResult] = []
        for line in lines.values():
            full_text = " ".join(line["words"]).strip()
            if not full_text:
                continue
            avg_conf = sum(line["confs"]) / len(line["confs"])
            bbox = BoundingBox(line["x1"], line["y1"], line["x2"], line["y2"])
            out.append(OCRResult(text=full_text, bbox=bbox, confidence=avg_conf))

        return out


def create_ocr_engine(backend: str = "auto", max_edge: Optional[int] = None) -> BaseOCREngine:
    """
    Factory function to initialize the preferred OCR backend.
    
    :param backend: 'auto', 'rapidocr', or 'tesseract'
    :param max_edge: Optional edge limit for RapidOCR downscaling
    """
    target = backend.lower().strip()

    if target in ("auto", "rapidocr"):
        try:
            return RapidOCREngine(max_edge=max_edge)
        except Exception as e:
            if target == "rapidocr":
                raise e

    if target in ("auto", "tesseract"):
        try:
            return TesseractOCREngine()
        except Exception as e:
            if target == "tesseract":
                raise e

    raise RuntimeError(
        "No working OCR engine found. Please install rapidocr-onnxruntime:\n"
        "pip install rapidocr-onnxruntime==1.2.3"
    )
