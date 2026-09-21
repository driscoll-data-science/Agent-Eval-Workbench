"""Tiny slug helper used by the example suite."""

import re


def slugify(text: str) -> str:
    """Lowercase and join words with '-'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words)
