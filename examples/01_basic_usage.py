"""Example 01: Basic screen recognition and querying."""

import sys
from pathlib import Path

# Add parent directory to sys.path so it works without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from screen_recognizer import ScreenRecognizer


def main():
    print("Initializing ScreenRecognizer (Local OCR + CV)...")
    recognizer = ScreenRecognizer()

    print("Recognizing active screen...")
    # Capture live screen and recognize
    state = recognizer.recognize()

    print(f"\nScreen Resolution: {state.width}x{state.height}")
    print(f"Total Detected Elements: {len(state.elements)}")
    print(f"Interactive Elements: {len(state.get_clickable())}\n")

    # 1. Print first 10 elements
    print("--- First 10 Elements ---")
    for el in state.elements[:10]:
        print(f"[{el.id}] {el.type.value:<8} | Text: '{el.text}' | Center: ({el.center_x}, {el.center_y})")

    # 2. Search by text (fuzzy & substring matching)
    search_query = "File"
    matches = state.find_by_text(search_query)
    print(f"\n--- Search for '{search_query}' ({len(matches)} matches) ---")
    for el in matches:
        print(f"Found: [{el.id}] '{el.text}' at ({el.center_x}, {el.center_y})")

    # 3. Save Set-of-Marks annotated visualization
    output_image = Path(__file__).parent / "screen_marked.png"
    recognizer.annotate(state, output_path=str(output_image), only_interactive=True)
    print(f"\nAnnotated visualization saved to: {output_image}")


if __name__ == "__main__":
    main()
