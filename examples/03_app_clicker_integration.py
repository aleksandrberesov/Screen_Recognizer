"""Example 03: Downstream App Clicker integration.

Demonstrates how an automation tool (such as App_Clicker or an RPA agent)
imports and uses ScreenRecognizer as its local vision module to locate buttons
by text and execute mouse clicks.
"""

import sys
import time
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from screen_recognizer import ScreenRecognizer


class LocalAppClicker:
    """A sample clicker that uses ScreenRecognizer for zero-LLM visual targeting."""

    def __init__(self):
        self.recognizer = ScreenRecognizer()

    def click_text(self, text: str, source=None, dry_run: bool = True) -> bool:
        """
        Locates an on-screen element matching `text` and clicks its center coordinate.
        """
        print(f"Scanning UI for text: '{text}'...")
        state = self.recognizer.recognize(source=source)

        matches = state.find_by_text(text)
        if not matches:
            print(f"[-] Text '{text}' not found on screen.")
            return False

        target = matches[0]
        x, y = target.center_x, target.center_y
        print(f"[+] Found [{target.id}] '{target.text}' ({target.type.value}) at pixel coordinates ({x}, {y})")

        if dry_run:
            print(f"[DRY-RUN] Would click mouse at ({x}, {y})")
            return True

        # Real click using ctypes / pyautogui
        try:
            import pyautogui
            pyautogui.click(x, y)
            print(f"[SUCCESS] Clicked at ({x}, {y})")
            return True
        except ImportError:
            # Fallback to Windows ctypes SendInput / mouse_event
            import ctypes
            ctypes.windll.user32.SetCursorPos(x, y)
            ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
            time.sleep(0.05)
            ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
            print(f"[SUCCESS] Clicked via Windows API at ({x}, {y})")
            return True


def main():
    clicker = LocalAppClicker()
    sample_file = Path(__file__).parent.parent / "tests" / "sample_app.png"
    source = sample_file if sample_file.exists() else None

    # Test locating any text (dry-run mode so it doesn't disrupt user's mouse)
    test_label = "Sign In"
    clicker.click_text(test_label, source=source, dry_run=True)


if __name__ == "__main__":
    main()
