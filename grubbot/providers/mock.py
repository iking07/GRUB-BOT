import json
import re
from typing import Dict, List

from .base import BaseProvider


class MockProvider(BaseProvider):
    """Deterministic provider used for local smoke tests without API keys."""

    def _extract_count(self, prompt: str, default: int = 5) -> int:
        match = re.search(r"generate\s+(\d+)", prompt.lower())
        if match:
            return int(match.group(1))
        return default

    def _extract_tool_name(self, prompt: str, default: str = "mock_tool") -> str:
        match = re.search(r"tool\s+name\s*:\s*([a-zA-Z0-9_\-]+)", prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1)

        # Targeted prompts contain expected tool snippets.
        json_name = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_\-]+)"', prompt)
        if json_name:
            return json_name.group(1)

        return default

    def _extract_parameters(self, prompt: str) -> List[Dict[str, str]]:
        params = []
        for line in prompt.splitlines():
            match = re.search(r"-\s*([a-zA-Z0-9_]+)\s*\(([^,\)]+),\s*(required|optional)\)", line.strip(), flags=re.IGNORECASE)
            if not match:
                continue
            params.append(
                {
                    "name": match.group(1),
                    "type": match.group(2).strip(),
                    "required": match.group(3).lower() == "required",
                }
            )
        return params

    def generate(self, prompt: str, system: str = "") -> str:
        count = self._extract_count(prompt)
        tool_name = self._extract_tool_name(prompt)
        params = self._extract_parameters(prompt)

        if not params:
            # Conservative fallback for targeted prompts where parameter lines are absent.
            params = [{"name": "query", "type": "string", "required": True}]

        items = []
        for i in range(count):
            args = {}
            for p in params:
                if (not p["required"]) and i % 3 == 0:
                    continue
                args[p["name"]] = f"{p['name']}_{i}"

            items.append(
                {
                    "user_query": f"Use {tool_name} request {i}",
                    "expected_tool_call": {
                        "name": tool_name,
                        "arguments": args,
                    },
                }
            )

        return json.dumps(items)
