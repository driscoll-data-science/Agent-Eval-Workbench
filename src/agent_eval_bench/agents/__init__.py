"""Agents under test that ship with the kit.

Only the minimal reference Agent SDK agent lives here; it exists so the SDK adapter path is
exercised end to end. Agents J builds next are separate packages that expose the same
entrypoint signature and are selected per profile via ``params.entrypoint``.
"""
