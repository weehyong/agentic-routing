"""
eddy.runtime.queue
==================
Work queue: FIFO with re-enqueue (DESIGN §4, §11).

The router pops the head, routes one step, and either emits the item or
re-enqueues it for its next step.  ``operator_backlog`` exposes a per-operator
backlog estimate (count of queued items currently eligible for each operator),
which P1 backpressure reads as ``q_a``.
"""
from __future__ import annotations

from collections import deque

from eddy.core.workitem import WorkItem


class WorkQueue:
    def __init__(self) -> None:
        self._q: deque[WorkItem] = deque()

    def push(self, item: WorkItem) -> None:
        self._q.append(item)

    def pop(self) -> WorkItem:
        return self._q.popleft()

    def empty(self) -> bool:
        return not self._q

    def __len__(self) -> int:
        return len(self._q)

    def operator_backlog(self, registry) -> dict[str, int]:
        """q_a = number of queued items currently eligible for operator a."""
        counts: dict[str, int] = {op.operator_id: 0 for op in registry.operators}
        for item in self._q:
            for i, op in enumerate(registry.operators):
                if i < len(item.ready) and item.ready[i] == 1 and (
                    i >= len(item.done) or item.done[i] == 0
                ):
                    counts[op.operator_id] += 1
        return counts
