"""Minimal reference Agent SDK agent: the SDK adapter's default entrypoint.

Yields every message from ``claude_agent_sdk.query`` unchanged so the adapter sees the raw
stream. A real agent keeps this signature and may add its own tools, hooks, or system prompt
on top of the options the adapter built from the profile.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from claude_agent_sdk import ClaudeAgentOptions, Message, query


async def run(prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Message]:
    async for message in query(prompt=prompt, options=options):
        yield message
