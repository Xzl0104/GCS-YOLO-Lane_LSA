"""Adapter boundary for low-level Codex spawn_agent calls.

The real ``spawn_agent`` callable is provided by the host Codex UI/tool layer.
Project code should call through this adapter so UI/schema defaults are cleaned
immediately before the runtime call is sent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from gcs_spawn_payload import normalize_spawn_payload


SpawnAgentCallable = Callable[..., Any]


def spawn_agent_with_normalized_payload(spawn_agent: SpawnAgentCallable, raw_payload: Mapping[str, Any]) -> Any:
    """Normalize a raw payload at the final boundary, then call spawn_agent."""

    payload = normalize_spawn_payload(raw_payload)
    return spawn_agent(**payload)
