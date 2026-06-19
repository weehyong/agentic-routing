"""
eddy.core.profile
=================
Core QCTR vector and per-operator EWMA profile.

Q is *maximized*; C, T, R are *minimized*.  as_tuple_for_dominance() negates Q so
every dimension is "lower is better" — the uniform convention used throughout
dominance.py and maintain.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QCTR:
    """Expected reward dimensions for one operator.

    Attributes
    ----------
    Q : float
        Quality / utility, higher is better, in [0, 1].
    C : float
        Cost (dollars + token-normalised), lower is better.
    T : float
        Latency (seconds), lower is better.
    R : float
        Risk (impact × p_fail), lower is better, in [0, 1].
    """

    Q: float
    C: float
    T: float
    R: float

    def as_tuple_for_dominance(self) -> tuple[float, float, float, float]:
        """Negate Q so ALL dims become 'lower is better' for a uniform comparison."""
        return (-self.Q, self.C, self.T, self.R)

    def __repr__(self) -> str:  # prettier for test output
        return f"QCTR(Q={self.Q:.4f}, C={self.C:.4f}, T={self.T:.4f}, R={self.R:.4f})"


@dataclass(slots=True)
class OperatorProfile:
    """Per-operator EWMA estimate of (Q, C, T, R), updated after each invocation.

    The EWMA tracks the current best estimate; n_invocations and the Welford
    accumulators (reward_mean, reward_m2) support the P3 contextual bandit.
    """

    operator_id: str
    est: QCTR
    alpha: float = 0.3          # EWMA smoothing factor  (λ in ML literature)
    n_invocations: int = 0
    reward_mean: float = 0.0    # Welford running mean of scalar rewards
    reward_m2: float = 0.0      # Welford running sum-of-squared-deviations

    def update_estimate(self, obs: QCTR) -> None:
        """Apply one EWMA step: est ← (1−α)·est + α·obs."""
        a, e = self.alpha, self.est
        self.est = QCTR(
            Q=(1 - a) * e.Q + a * obs.Q,
            C=(1 - a) * e.C + a * obs.C,
            T=(1 - a) * e.T + a * obs.T,
            R=(1 - a) * e.R + a * obs.R,
        )
        self.n_invocations += 1

    @classmethod
    def from_qctr(cls, operator_id: str, est: QCTR, alpha: float = 0.3) -> "OperatorProfile":
        """Convenience constructor."""
        return cls(operator_id=operator_id, est=est, alpha=alpha)
