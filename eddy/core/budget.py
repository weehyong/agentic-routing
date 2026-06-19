"""
eddy.core.budget
================
Budget tracker with dynamic β (DESIGN §5.5).

β rises as the token budget depletes, so the ``C`` dimension the router sees for
token-heavy operators grows and they leave the skyline — "the frontier shifts
toward cheaper operators as budget depletes" becomes a mechanical consequence of
``cost(...)``, not an assertion (DESIGN §12 step 4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BudgetTracker:
    dollar_budget: float
    token_budget: int
    dollars_spent: float = 0.0
    tokens_spent: int = 0
    alpha: float = 1.0
    beta_floor: float = 0.5

    @property
    def tokens_remaining(self) -> int:
        return max(1, self.token_budget - self.tokens_spent)

    @property
    def beta(self) -> float:
        """β rises as budget depletes: frontier shifts toward cheaper operators."""
        return self.beta_floor / max(1e-3, self.tokens_remaining / self.token_budget)

    def cost(self, dollars: float, tokens: int) -> float:
        """Budget-aware cost used to recompute a candidate's C dimension at routing time."""
        return self.alpha * dollars + self.beta * (tokens / self.tokens_remaining)

    def charge(self, dollars: float, tokens: int) -> None:
        self.dollars_spent += dollars
        self.tokens_spent += tokens

    def exhausted(self) -> bool:
        return self.dollars_spent >= self.dollar_budget or self.tokens_spent >= self.token_budget
