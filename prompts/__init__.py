"""Prompt expansion package."""
from .templates import (
    ZERO_SHOT,
    FEW_SHOT,
    ZERO_SHOT_PROMPT_TEMPLATE,
    FEW_SHOT_PROMPT_TEMPLATE,
    format_expansion_prompt,
)

__all__ = [
    "ZERO_SHOT",
    "FEW_SHOT",
    "ZERO_SHOT_PROMPT_TEMPLATE",
    "FEW_SHOT_PROMPT_TEMPLATE",
    "format_expansion_prompt",
]
