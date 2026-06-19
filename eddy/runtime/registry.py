"""
eddy.runtime.registry
=====================
Operator registry: k operators, a stable index map, the shared profile map
(DESIGN §4, §11).

Masks (``ready`` / ``done`` / ``required``) are length-``k`` lists aligned to this
index, so eligibility and dominance stay cheap and the invariants are trivially
checkable.
"""
from __future__ import annotations

from eddy.core.operator import Operator
from eddy.core.profile import OperatorProfile


class OperatorRegistry:
    def __init__(self, operators: list[Operator]) -> None:
        self.operators: list[Operator] = list(operators)
        self._index: dict[str, int] = {op.operator_id: i for i, op in enumerate(self.operators)}
        if len(self._index) != len(self.operators):
            raise ValueError("duplicate operator_id in registry")
        # Shared profile map — skyline maintainer and policies read/update this.
        self.profiles: dict[str, OperatorProfile] = {
            op.operator_id: op.profile for op in self.operators
        }

    def __len__(self) -> int:
        return len(self.operators)

    def index(self, operator_id: str) -> int | None:
        return self._index.get(operator_id)

    def get(self, operator_id: str) -> Operator:
        return self.operators[self._index[operator_id]]

    @property
    def ids(self) -> list[str]:
        return [op.operator_id for op in self.operators]

    def eligible_ids(self, item) -> list[str]:
        """Operators with ready=1 ∧ done=0 for this item (E(w))."""
        out: list[str] = []
        ready, done = item.ready, item.done
        for i, op in enumerate(self.operators):
            if i < len(ready) and ready[i] == 1 and (i >= len(done) or done[i] == 0):
                out.append(op.operator_id)
        return out

    def new_done_mask(self) -> list[int]:
        return [0] * len(self.operators)

    def required_mask(self, required_ids: list[str]) -> list[int]:
        mask = [0] * len(self.operators)
        for rid in required_ids:
            idx = self._index.get(rid)
            if idx is not None:
                mask[idx] = 1
        return mask
