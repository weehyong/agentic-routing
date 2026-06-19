"""
eddy.core.ids
=============
Monotonic id and replay-id generation (DESIGN §5, §4 ``core/ids.py``).

Ids are generated from a process-local monotonic counter rather than wall-clock
time or randomness, so a seeded run is exactly reproducible and every id is
unique within the run.  ``replay_id`` tags a single routing decision so the
decision (and the invocation it triggers) can be deterministically re-executed
under the seeded ``rng`` (DESIGN §10, §14).
"""
from __future__ import annotations

import itertools

_counter = itertools.count()


def new_id(prefix: str = "id") -> str:
    """Return a fresh monotonic id of the form ``<prefix>-<n>``."""
    return f"{prefix}-{next(_counter)}"


def new_replay_id() -> str:
    """Return a fresh replay id for one routing decision."""
    return new_id("rp")


def reset() -> None:
    """Reset the global counter.  For deterministic test setup only."""
    global _counter
    _counter = itertools.count()
