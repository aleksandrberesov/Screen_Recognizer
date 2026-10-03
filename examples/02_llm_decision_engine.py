"""Example 02: Using ScreenRecognizer with an LLM as the decision engine.

In this architecture:
  1. ScreenRecognizer runs 100% locally to parse visual elements, text, and coordinates.
  2. The LLM only receives a compact text list of elements (~200-400 tokens) and the user goal.
  3. The LLM outputs a simple JSON action: {"action": "click", "element_id": 3}.
  4. The tool maps element_id 3 to exact pixel coordinates (x, y) with zero coordinate hallucination.
"""

import json
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from screen_recognizer import PromptFormatter, ScreenRecognizer


def mock_llm_decision_engine(prompt: str) -> str:
    """
    Simulated LLM decision engine.
    In production, pass `prompt` to OpenAI, Anthropic, or a local model (Ollama / Llama / Qwen).
    """
    print("\n[LLM Decision Engine Received Prompt]:")
    print("-" * 50)
    print(prompt[:600] + ("\n... [truncated]" if len(prompt) > 600 else ""))
    print("-" * 50)

    # Simulated reasoning & action output
    return json.dumps({
        "thought": "Found the target interactive element matching the user's objective.",
        "action": "click",
        "element_id": 1,
    })


def main():
    recognizer = ScreenRecognizer()

    # Step 1: Local perception (zero LLM involved)
    sample_file = Path(__file__).parent.parent / "tests" / "sample_app.png"
    if sample_file.exists():
        print(f"1. Recognizing screenshot locally from {sample_file.name}...")
        state = recognizer.recognize(source=sample_file)
    else:
        print("1. Recognizing live screen locally...")
        state = recognizer.recognize()

    # Step 2: Format concise prompt for LLM decision engine
    goal = "Sign into the account using alex@automation.io"
    prompt = recognizer.to_prompt(state, goal=goal, only_interactive=True)

    # Step 3: LLM decides the action based on the structured state
    print("2. Querying LLM decision engine...")
    llm_response = mock_llm_decision_engine(prompt)

    # Step 4: Parse LLM output and resolve exact pixel coordinates
    action, target_element = PromptFormatter.parse_action_response(llm_response, state)

    print("\n3. Parsed Action Plan:")
    print(f"   Action Type : {action.get('action')}")
    print(f"   Target ID   : {action.get('element_id')}")

    if target_element:
        print(f"   Resolved UI : [{target_element.id}] {target_element.type.value} '{target_element.text}'")
        print(f"   Exact Target: Screen Coordinate ({target_element.center_x}, {target_element.center_y})")
        print(f"   Bounding Box: {target_element.bbox.to_tuple()}")
        print("\n   Ready for App Clicker to execute click at exact coordinates!")
    else:
        print("   Element ID not found in screen state.")


if __name__ == "__main__":
    main()
