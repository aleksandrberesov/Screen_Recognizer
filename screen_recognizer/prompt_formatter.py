"""Prompt generation utilities for feeding recognized screen states into LLM decision engines."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .models import ScreenState, UIElement


class PromptFormatter:
    """Formats ScreenState into token-efficient prompts for LLM planners."""

    @staticmethod
    def format_for_llm(
        state: ScreenState,
        goal: Optional[str] = None,
        only_interactive: bool = True,
    ) -> str:
        """
        Generate a concise, LLM-ready prompt containing the screen state and instructions.
        """
        elements = state.get_clickable() if only_interactive else state.elements
        lines = [
            f"Screen Resolution: {state.width}x{state.height}",
            f"Available Interactive Elements ({len(elements)} items):",
        ]

        for el in elements:
            txt = f" \"{el.text}\"" if el.text else ""
            lines.append(
                f"- [{el.id}] {el.type.value.upper()}{txt} | Center: ({el.center_x}, {el.center_y})"
            )

        if goal:
            lines.extend([
                "",
                f"Current Goal: {goal}",
                "Respond with the next action in JSON format, for example:",
                '{"action": "click", "element_id": 0}',
                '{"action": "type", "element_id": 1, "text": "hello"}',
                '{"action": "wait", "seconds": 2}',
                '{"action": "finish", "status": "success"}',
            ])

        return "\n".join(lines)

    @staticmethod
    def parse_action_response(response_text: str, state: ScreenState) -> tuple[dict[str, Any], Optional[UIElement]]:
        """
        Parse an action JSON from an LLM's text response and resolve the target UIElement.
        
        Returns (action_dict, resolved_ui_element).
        """
        # Try finding JSON block in markdown ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        raw_json = match.group(1) if match else None

        if not raw_json:
            # Fallback to finding any { ... } block
            match = re.search(r"(\{.*?\})", response_text, re.DOTALL)
            raw_json = match.group(1) if match else None

        if not raw_json:
            raise ValueError(f"Could not find valid JSON action in model response: {response_text[:200]}")

        action = json.loads(raw_json)
        element = None
        if "element_id" in action:
            element = state.find_by_id(int(action["element_id"]))

        return action, element
