"""
eddy.governance.permissions
============================
Session-scoped permission scope and allowlists (DESIGN §8).

A PermissionScope holds the capabilities the current session was granted.  It is
the static governance input the gate chain reads; the per-item mutable state is
the ``done`` mask alone (DESIGN §8 "Where things live").
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PermissionScope:
    granted: set[str] = field(default_factory=set)              # scopes the session holds
    tool_allowlist: set[str] = field(default_factory=set)
    domain_allowlist: set[str] = field(default_factory=set)
    path_allowlist: set[str] = field(default_factory=set)
    transmit_allowed_classes: set[str] = field(
        default_factory=lambda: {"public", "internal"}
    )
