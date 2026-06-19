"""
eddy.core.workitem
==================
Work item, status, signals, and the ready/done/required masks (DESIGN §5.2).

A WorkItem is the orchestration analogue of a tuple.  ``ready`` / ``done`` /
``required`` are length-``k`` 0/1 lists aligned to the OperatorRegistry's index
map, so eligibility (Invariant 1) and completion (Invariant 3) are trivially
checkable.  ``required`` decouples "this operator ran" from "this operator was
required before emission".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ItemStatus(str, Enum):
    ACTIVE = "active"
    EMITTED = "emitted"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass(slots=True)
class Signals:
    """Per-item evolving evidence: quality sub-scores, metadata, confidence."""

    quality: dict[str, float] = field(default_factory=dict)   # {"completeness": 0.8}
    metadata: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(slots=True)
class WorkItem:
    item_id: str
    task_type: str                                  # "schema_profile", "record_match", ...
    payload: dict[str, object] = field(default_factory=dict)
    signals: Signals = field(default_factory=Signals)
    ready: list[int] = field(default_factory=list)      # ready(w) ∈ {0,1}^k
    done: list[int] = field(default_factory=list)       # done(w)  ∈ {0,1}^k
    required: list[int] = field(default_factory=list)   # steps required for φ
    status: ItemStatus = ItemStatus.ACTIVE
    lineage_id: str = ""
    parent_id: str | None = None
    sla_tier: str = "standard"                      # routing context
    safety_tier: str = "normal"
    data_classification: str = "internal"           # governance-gate input
