"""Screen and window capture utilities supporting PIL ImageGrab, UIAutomation, and MSS."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageGrab

from .models import BoundingBox


class ScreenCapture:
    """Capture screenshots from screens, specific bounding boxes, or window handles."""

    @staticmethod
    def capture_screen(
        bbox: Optional[Union[BoundingBox, Sequence[int], tuple[int, int, int, int]]] = None,
        all_screens: bool = True,
    ) -> Image.Image:
        """
        Capture the entire screen or a sub-region bounding box.
        Tries PIL ImageGrab -> UIAutomation (native Windows) -> MSS.
        Returns a PIL Image in RGB mode.
        """
        box_tuple = None
        if bbox is not None:
            if isinstance(bbox, BoundingBox):
                box_tuple = bbox.to_tuple()
            else:
                box_tuple = tuple(int(v) for v in bbox[:4])

        # 1. Attempt PIL ImageGrab
        try:
            img = ImageGrab.grab(bbox=box_tuple, all_screens=all_screens)
            return img.convert("RGB")
        except Exception:
            pass

        # 2. Attempt Windows UIAutomation (Native Windows capture, works in remote/restricted stations)
        try:
            import uiautomation as auto
            root = auto.GetRootControl()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                success = root.CaptureToImage(tmp_path)
                if success and os.path.isfile(tmp_path):
                    full_img = Image.open(tmp_path).convert("RGB")
                    if box_tuple is not None:
                        return full_img.crop(box_tuple)
                    return full_img
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        except Exception:
            pass

        # 3. Attempt MSS
        try:
            import mss
            with mss.MSS() as sct:
                if box_tuple is not None:
                    monitor = {
                        "left": box_tuple[0],
                        "top": box_tuple[1],
                        "width": box_tuple[2] - box_tuple[0],
                        "height": box_tuple[3] - box_tuple[1],
                    }
                    sct_img = sct.grab(monitor)
                else:
                    monitor = sct.monitors[0 if all_screens else 1]
                    sct_img = sct.grab(monitor)
                return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception:
            pass

        raise RuntimeError(
            "Failed to capture screen. PIL ImageGrab, UIAutomation, and MSS all failed."
        )

    @staticmethod
    def screen_origin(
        bbox: Optional[Union[BoundingBox, Sequence[int]]] = None,
        all_screens: bool = True,
    ) -> tuple[int, int]:
        """
        Absolute screen position of pixel (0, 0) in an image from capture_screen().
        Non-zero for region captures, or when a monitor sits left of or above the primary one.
        """
        if bbox is not None:
            box = bbox if isinstance(bbox, BoundingBox) else BoundingBox.from_ltrb(*bbox[:4])
            return box.x1, box.y1
        if not all_screens or sys.platform != "win32":
            return 0, 0

        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
            user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            # Query in physical pixels, as ImageGrab captures them (-4 = per-monitor aware v2)
            previous = user32.SetThreadDpiAwarenessContext(-4)
            try:
                return user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)  # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN
            finally:
                if previous:
                    user32.SetThreadDpiAwarenessContext(previous)
        except Exception:
            return 0, 0

    @staticmethod
    def capture_window(
        window_title: Optional[str] = None,
        hwnd: Optional[int] = None,
    ) -> tuple[Image.Image, BoundingBox]:
        """
        Capture a specific application window by window title (substring) or HWND handle.
        Returns (PIL Image, BoundingBox of window on screen).
        """
        # Try UIAutomation first for robust window binding
        try:
            import uiautomation as auto

            if hwnd:
                ctrl = auto.ControlFromHandle(hwnd)
            elif window_title:
                ctrl = auto.WindowControl(searchDepth=1, SubName=window_title)
            else:
                raise ValueError("Must specify window_title or hwnd.")

            if not ctrl.Exists(maxSearchSeconds=2):
                raise RuntimeError(f"Window matching '{window_title or hwnd}' not found.")

            rect = ctrl.BoundingRectangle
            bbox = BoundingBox(rect.left, rect.top, rect.right, rect.bottom)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                success = ctrl.CaptureToImage(tmp_path)
                if success and os.path.isfile(tmp_path):
                    img = Image.open(tmp_path).convert("RGB")
                    return img, bbox
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            # If CaptureToImage failed, fall back to crop from screen
            return ScreenCapture.capture_screen(bbox=bbox), bbox

        except Exception as e:
            raise RuntimeError(f"Failed to capture window '{window_title or hwnd}': {e}")

    @staticmethod
    def load_image(source: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load an image from filepath, numpy array, or existing PIL Image into RGB PIL format.
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, np.ndarray):
            if len(source.shape) == 2:
                return Image.fromarray(source).convert("RGB")
            elif source.shape[2] == 4:
                return Image.fromarray(source, mode="RGBA").convert("RGB")
            else:
                return Image.fromarray(source).convert("RGB")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")
