"""Utilities for preparing low-level Codex spawn_agent payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PAYLOAD_XOR_ERROR = "spawn_agent payload must contain exactly one non-empty payload field: message or items"


def _is_empty_value(value: Any) -> bool:
    """Return True for values that must not be serialized into spawn_agent."""

    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, list):
        return len(value) == 0
    return False


def normalize_spawn_payload(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop empty fields and enforce the message/items payload XOR contract.

    This must run at the final adapter boundary, after any UI/schema layer has
    applied defaults. Some call surfaces add ``items: []`` to message-only
    payloads; the runtime still treats that empty array as a present payload
    field, so it must be removed before the spawn_agent call is sent.
    """

    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if _is_empty_value(value):
            continue
        cleaned[key] = value

    message = cleaned.get("message")
    if message is not None and not isinstance(message, str):
        raise ValueError("spawn_agent message must be a string")
    if isinstance(message, str) and not message.strip():
        cleaned.pop("message", None)

    items = cleaned.get("items")
    if items is not None and not isinstance(items, list):
        raise ValueError("spawn_agent items must be a list")
    if isinstance(items, list) and not items:
        cleaned.pop("items", None)

    has_message = "message" in cleaned
    has_items = "items" in cleaned
    if has_message == has_items:
        raise ValueError(PAYLOAD_XOR_ERROR)

    return cleaned
