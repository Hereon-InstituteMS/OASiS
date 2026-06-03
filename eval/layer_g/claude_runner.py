"""Layer G Claude runner — call Anthropic API with ablation-modulated catalog.

Reads ANTHROPIC_API_KEY from environment. Falls back to anthropic.Anthropic()
default credential resolution if unset.

Public entry point:
  run_claude(task, mode, knowledge) -> ClaudeResponse

`knowledge` is the (possibly ablated) dict from ablation_modes.apply_ablation.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from anthropic import Anthropic
except ImportError as e:
    raise SystemExit(
        "anthropic SDK not installed. Run: "
        "pip install anthropic"
    ) from e


CLAUDE_MODEL = os.environ.get("LAYER_G_CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 4096


@dataclass
class ClaudeResponse:
    response_id: str
    text: str
    code_blocks: list[str]
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


def build_system_prompt(backend: str, physics: str, knowledge: dict[str, Any]) -> str:
    """Render a system prompt that includes the (ablated) catalog as
    structured knowledge the model should condition on."""
    parts = [
        f"You are an expert simulation engineer using the {backend} FEM "
        f"backend to solve {physics} problems. You will be given a task "
        f"and a snapshot of {backend}'s known pitfalls and templates. "
        f"Use only what is provided here — do not invent functions "
        f"or features not present in the catalog.\n",
    ]

    if template := knowledge.get("template"):
        parts.append("== Template (starting point) ==\n")
        parts.append(template if isinstance(template, str) else json.dumps(template, indent=2))
        parts.append("\n")

    if signatures := knowledge.get("signatures"):
        parts.append("== Known signatures ==\n")
        parts.append(json.dumps(signatures, indent=2))
        parts.append("\n")

    pitfalls = knowledge.get("pitfalls") or []
    parts.append(f"== Pitfalls ({len(pitfalls)} entries) ==\n")
    if not pitfalls:
        parts.append("(none provided)\n")
    else:
        for i, p in enumerate(pitfalls, 1):
            parts.append(f"--- Pitfall {i} ---\n")
            if isinstance(p, dict):
                parts.append(json.dumps(p, indent=2))
            else:
                parts.append(str(p))
            parts.append("\n")

    parts.append(
        "\n== Output format ==\n"
        "Respond with a SINGLE fenced code block containing the complete "
        "runnable solution. No prose around it. The code must be self-"
        "contained and runnable from a clean working directory."
    )
    return "".join(parts)


def build_user_message(task: dict[str, Any]) -> str:
    return task["prompt"]


def run_claude(
    task: dict[str, Any],
    mode: str,
    knowledge: dict[str, Any],
    client: Anthropic | None = None,
) -> ClaudeResponse:
    """One round-trip Claude call.

    `mode` is informational (carried through for downstream logging); the
    catalog ablation has already been applied to `knowledge` by the caller.
    """
    _ = mode
    api = client if client is not None else Anthropic()

    system = build_system_prompt(task["backend"], task["physics"], knowledge)
    user = build_user_message(task)

    msg = api.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = "".join(b.text for b in msg.content if b.type == "text")
    code_blocks = _extract_code_blocks(text)

    return ClaudeResponse(
        response_id=msg.id,
        text=text,
        code_blocks=code_blocks,
        stop_reason=msg.stop_reason,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
    )


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)?\n(.*?)```", re.DOTALL)


def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks. If no fence, treat entire text as one
    code block (Claude sometimes omits fences for short snippets)."""
    blocks = _FENCE_RE.findall(text)
    if not blocks and text.strip():
        return [text.strip()]
    return [b.strip() for b in blocks if b.strip()]
