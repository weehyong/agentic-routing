"""
eddy.observability.frontier_log
===============================
Pareto-frontier evolution artifact (DESIGN §10, RQ6).

Each tick the runtime snapshots the skyline: which operators are members, their
current ``(Q,C,T,R)`` estimates, and which was selected.  ``dump()`` is the
governance/monitoring artifact — it makes "the frontier shifts toward cheaper
operators as budget depletes" *observable*, not merely asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eddy.core.profile import OperatorProfile


@dataclass(frozen=True, slots=True)
class FrontierSnapshot:
    tick: int
    item_id: str
    members: tuple[str, ...]
    estimates: dict[str, tuple[float, float, float, float]]
    selected: str


@dataclass(slots=True)
class FrontierLog:
    snapshots: list[FrontierSnapshot] = field(default_factory=list)

    def record(
        self,
        tick: int,
        item_id: str,
        members: list[str],
        profiles: dict[str, OperatorProfile],
        selected: str,
    ) -> None:
        ests = {
            m: (profiles[m].est.Q, profiles[m].est.C, profiles[m].est.T, profiles[m].est.R)
            for m in members
        }
        self.snapshots.append(
            FrontierSnapshot(tick, item_id, tuple(members), ests, selected)
        )

    def dump(self) -> list[dict]:
        return [
            {
                "tick": s.tick,
                "item_id": s.item_id,
                "members": list(s.members),
                "estimates": s.estimates,
                "selected": s.selected,
            }
            for s in self.snapshots
        ]

    def members_over_time(self, operator_id: str) -> list[bool]:
        """Membership trace for one operator — e.g. to assert it leaves the skyline."""
        return [operator_id in s.members for s in self.snapshots]
