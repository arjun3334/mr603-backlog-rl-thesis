"""
backlog_rl.env.generator
------------------------
Synthetic PulseBoard-style backlog generator.

Produces backlog items following the feature schema defined in the
MR603 Week 5 MDP Specification (state_space Section 1):
    business_value, effort, risk, dependency_count,
    demand_signal, urgency, age

Distributions are chosen deliberately (see Chapter 3, Section 3.3.1)
to avoid a trivially separable / uniform toy dataset.
"""

import numpy as np
from dataclasses import dataclass, field, asdict

FIBONACCI_POINTS = [1, 2, 3, 5, 8, 13]
MAX_EFFORT = max(FIBONACCI_POINTS)

# A small pool of realistic PulseBoard-style item names, used only to
# make generated backlogs readable in demos / screenshots. Does not
# affect the underlying numeric features used by the policy.
ITEM_NAME_POOL = [
    "Bulk CSV export", "SSO login (SAML)", "Dark mode UI", "Async job queue",
    "Slack integration", "Custom dashboard widgets", "Audit log export",
    "Two-factor authentication", "Webhook retries", "API rate limiting",
    "Kanban swimlanes", "Time-tracking report", "Mobile push notifications",
    "Bulk item archive", "Custom fields for tickets", "SLA breach alerts",
    "Guest viewer role", "Recurring task templates", "Dependency graph view",
    "CSV import validation", "Public roadmap page", "Sprint burndown chart",
    "Email digest summaries", "Attachment virus scanning", "Search autocomplete",
    "Bulk reassign owner", "Custom status workflows", "Team capacity planner",
    "Item history timeline", "Keyboard shortcuts", "Offline mode (PWA)",
    "Multi-currency billing", "Usage analytics dashboard", "SSO for SCIM provisioning",
    "Comment @mentions", "Item templates library", "Read-only API tokens",
    "Data residency (EU region)", "Custom SLA policies", "Bulk label editor",
]


@dataclass
class BacklogItem:
    item_id: int
    name: str
    business_value: float      # [0, 1]
    effort_raw: int            # Fibonacci points {1,2,3,5,8,13}
    effort: float              # normalised [0, 1]
    risk: float                # [0, 1]
    dependency_count: int      # >= 0, capped at 3
    demand_signal: float       # [0, 1]
    urgency: float             # [0, 1]
    age: int = 0               # weeks in backlog, increments each cycle

    def to_feature_vector(self):
        """Return the 7-feature vector used by the MDP state (Week 5 spec, Section 1.1)."""
        return np.array([
            self.business_value,
            self.effort,
            self.risk,
            min(self.dependency_count, 3) / 3.0,  # normalised for the observation
            self.demand_signal,
            self.urgency,
            min(self.age, 20) / 20.0,             # normalised, capped for stability
        ], dtype=np.float32)


class BacklogGenerator:
    """
    Generates realistic synthetic backlog items per Chapter 3, Section 3.3.1.

    Distributions (deliberately non-uniform, see thesis Chapter 3):
        business_value      ~ Beta(2, 2)     symmetric, moderately peaked
        effort               ~ discrete draw from Fibonacci set
        risk                 ~ Beta(1.5, 3)  right-skewed (mostly low/moderate)
        dependency_count     ~ Poisson(0.6), capped at 3
        demand_signal        ~ Beta(2, 3)
        urgency               ~ Beta(1.5, 4)  mostly non-urgent
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._next_id = 0
        self._name_cycle = list(ITEM_NAME_POOL)
        self.rng.shuffle(self._name_cycle)
        self._name_idx = 0

    def _next_name(self):
        name = self._name_cycle[self._name_idx % len(self._name_cycle)]
        self._name_idx += 1
        return name

    def generate_item(self) -> BacklogItem:
        item_id = self._next_id
        self._next_id += 1

        business_value = float(self.rng.beta(2, 2))
        effort_raw = int(self.rng.choice(FIBONACCI_POINTS))
        effort = effort_raw / MAX_EFFORT
        risk = float(self.rng.beta(1.5, 3))
        dependency_count = int(min(self.rng.poisson(0.6), 3))
        demand_signal = float(self.rng.beta(2, 3))
        urgency = float(self.rng.beta(1.5, 4))

        return BacklogItem(
            item_id=item_id,
            name=self._next_name(),
            business_value=round(business_value, 3),
            effort_raw=effort_raw,
            effort=round(effort, 3),
            risk=round(risk, 3),
            dependency_count=dependency_count,
            demand_signal=round(demand_signal, 3),
            urgency=round(urgency, 3),
            age=0,
        )

    def generate_backlog(self, n_items: int):
        return [self.generate_item() for _ in range(n_items)]


if __name__ == "__main__":
    gen = BacklogGenerator(seed=42)
    backlog = gen.generate_backlog(8)
    print(f"{'Item':30s} {'Value':>6s} {'Effort':>7s} {'Risk':>6s} {'Deps':>5s} {'Demand':>7s} {'Urgency':>8s}")
    for it in backlog:
        print(f"{it.name:30s} {it.business_value:6.2f} {it.effort_raw:>4d}pt {it.risk:6.2f} "
              f"{it.dependency_count:5d} {it.demand_signal:7.2f} {it.urgency:8.2f}")
