"""
eddy.observability.events
=========================
Structured event schema (DESIGN §10).

Events are the chronological, governance-facing complement to the per-item
``LineageRecord``.  Each carries a ``replay_id`` so any decision can be
re-executed deterministically under the seeded ``rng``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    item_id: str
    replay_id: str
    ts: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)
