"""Generate a rich realistic desktop application screenshot and run ScreenRecognizer on it."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
from screen_recognizer import ScreenRecognizer


def generate_mock_app_image() -> Image.Image:
    """Create a realistic modern desktop UI (1000x700)."""
    img = Image.new("RGB", (1000, 700), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # 1. Title bar
    draw.rectangle([0, 0, 1000, 40], fill=(30, 41, 59))
    draw.text((20, 12), "Automation Dashboard v2.0 - [Production Environment]", fill=(255, 255, 255))
    # Window controls
    draw.rectangle([920, 10, 945, 30], fill=(71, 85, 105), outline=(100, 116, 139))
    draw.text((930, 12), "-", fill=(255, 255, 255))
    draw.rectangle([955, 10, 980, 30], fill=(220, 38, 38), outline=(185, 28, 28))
    draw.text((965, 12), "X", fill=(255, 255, 255))

    # 2. Navigation bar / Menus
    draw.rectangle([0, 40, 1000, 80], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.text((30, 54), "File", fill=(15, 23, 42))
    draw.text((80, 54), "Edit", fill=(15, 23, 42))
    draw.text((130, 54), "View", fill=(15, 23, 42))
    draw.text((185, 54), "Settings", fill=(15, 23, 42))
    draw.text((260, 54), "Help", fill=(15, 23, 42))

    # Search bar
    draw.rectangle([650, 48, 880, 72], fill=(241, 245, 249), outline=(203, 213, 225))
    draw.text((660, 53), "Search projects...", fill=(148, 163, 184))

    # 3. Form & Controls Area
    draw.rectangle([40, 110, 450, 380], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.text((60, 130), "User Authentication", fill=(15, 23, 42))

    # Input 1: Username
    draw.text((60, 170), "Username or Email:", fill=(71, 85, 105))
    draw.rectangle([60, 195, 420, 230], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.text((70, 205), "alex@automation.io", fill=(30, 41, 59))

    # Input 2: Password
    draw.text((60, 250), "Password:", fill=(71, 85, 105))
    draw.rectangle([60, 275, 420, 310], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.text((70, 285), "password123", fill=(30, 41, 59))

    # Buttons
    draw.rectangle([60, 330, 180, 365], fill=(37, 99, 235), outline=(29, 78, 216))
    draw.text((95, 340), "Sign In", fill=(255, 255, 255))

    draw.rectangle([200, 330, 320, 365], fill=(241, 245, 249), outline=(203, 213, 225))
    draw.text((235, 340), "Cancel", fill=(71, 85, 105))

    # 4. Action Cards Area
    draw.rectangle([490, 110, 950, 380], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.text((510, 130), "Quick Actions", fill=(15, 23, 42))

    draw.rectangle([510, 170, 710, 220], fill=(236, 253, 245), outline=(167, 243, 208))
    draw.text((540, 187), "Run Test Suite", fill=(5, 150, 105))

    draw.rectangle([730, 170, 930, 220], fill=(254, 242, 242), outline=(254, 202, 202))
    draw.text((760, 187), "Deploy Build", fill=(220, 38, 38))

    return img


def main():
    img = generate_mock_app_image()
    sample_path = Path(__file__).parent / "sample_app.png"
    img.save(sample_path)
    print(f"Generated realistic sample app screenshot: {sample_path}")

    recognizer = ScreenRecognizer()
    state = recognizer.recognize(source=img)

    print(f"\nTotal Elements Found : {len(state.elements)}")
    print(f"Interactive Elements : {len(state.get_clickable())}")

    print("\n--- Detected Elements ---")
    for el in state.elements:
        print(f"[{el.id:2d}] {el.type.value:<8} | \"{el.text}\" at ({el.center_x:3d}, {el.center_y:3d}) bbox={el.bbox.to_tuple()}")

    # Test Search
    print("\n--- Testing Element Search ---")
    sign_in = state.find_by_text("Sign In")
    assert len(sign_in) > 0, "Sign In button not found!"
    print(f"[OK] Found 'Sign In' at {sign_in[0].center}")

    run_tests = state.find_by_text("Run Test Suite")
    assert len(run_tests) > 0, "Run Test Suite button not found!"
    print(f"[OK] Found 'Run Test Suite' at {run_tests[0].center}")

    deploy = state.find_by_text("Deploy Build")
    assert len(deploy) > 0, "Deploy Build button not found!"
    print(f"[OK] Found 'Deploy Build' at {deploy[0].center}")

    # Generate Set-of-Marks annotated image
    annotated_path = Path(__file__).parent / "sample_annotated.png"
    recognizer.annotate(state, image=img, output_path=str(annotated_path), only_interactive=False)
    print(f"\n[OK] Set-of-Marks annotated visualization saved to: {annotated_path}")

    # Print LLM Markdown
    print("\n--- LLM Prompt View (Token Optimized) ---")
    print(state.to_markdown(only_interactive=True))


if __name__ == "__main__":
    main()
