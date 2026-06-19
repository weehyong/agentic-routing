"""
eddy.observability.sink
=======================
Event sinks (DESIGN §10).

``InMemorySink`` for tests/dashboards; ``JSONLSink`` for an immutable,
append-only audit log.  Both satisfy the ``EventSink`` Protocol so the runtime is
agnostic to where events land.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from eddy.observability.events import Event


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


@dataclass(slots=True)
class InMemorySink:
    events: list[Event] = field(default_factory=list)

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]


@dataclass(slots=True)
class JSONLSink:
    """Append-only JSONL audit log.  Opens in append mode and flushes per event."""

    path: str

    def emit(self, event: Event) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(event), default=str) + "\n")


@dataclass(slots=True)
class NullSink:
    """Discards events — for runs that need no observability overhead."""

    def emit(self, event: Event) -> None:
        return None
