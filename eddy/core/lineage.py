"""
eddy.core.lineage
=================
Append-only per-item lineage — Invariant 5 (DESIGN §5.4).

Every routing decision, invocation, tool call, mask update, emission, failure,
and gate evaluation is appended to ``L(w)`` with a monotonic ``seq`` and the
``replay_id`` of the decision that produced it.  The only public mutator is
``append`` — events are never edited or removed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class LineageKind(str, Enum):
    ROUTING_DECISION = "routing_decision"
    INVOCATION = "invocation"
    TOOL_CALL = "tool_call"
    MASK_UPDATE = "mask_update"
    EMISSION = "emission"
    FAILURE = "failure"
    GATE = "gate"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class LineageEvent:
    seq: int
    kind: LineageKind
    item_id: str
    replay_id: str
    ts: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)            # JSON-serializable


@dataclass(slots=True)
class LineageRecord:
    """Append-only ``L(w)``.  The only public mutator is ``append``."""

    item_id: str
    events: list[LineageEvent] = field(default_factory=list)
    _seq: int = 0

    def append(self, kind: LineageKind, replay_id: str, **detail) -> LineageEvent:
        ev = LineageEvent(self._seq, kind, self.item_id, replay_id, detail=detail)
        self.events.append(ev)
        self._seq += 1
        return ev

    def count(self, kind: LineageKind | None = None) -> int:
        if kind is None:
            return len(self.events)
        return sum(1 for e in self.events if e.kind == kind)
