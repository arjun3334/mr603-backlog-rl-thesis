"""
backlog_rl.tests.test_static_scorers
--------------------------------------
Validates RICE and WSJF implementations against hand-calculated
examples, as promised in the Week 8 Gantt entry ("implement RICE and
WSJF baseline scoring functions and unit-test them against
hand-calculated example backlogs").
"""

import unittest
from backlog_rl.env.generator import BacklogItem
from backlog_rl.baselines.static_scorers import rice_score, wsjf_score, assign_tiers, Tier


class TestRICEScore(unittest.TestCase):
    def test_hand_calculated_example(self):
        # Hand calculation:
        # Reach (demand_signal) = 0.60, Impact (business_value) = 0.80
        # risk = 0.30 (below 0.7 threshold -> confidence = 0.8)
        # effort = 0.40
        # RICE = (0.60 * 0.80 * 0.8) / 0.40 = 0.384 / 0.40 = 0.96
        item = BacklogItem(
            item_id=1, name="Test Item", business_value=0.80,
            effort_raw=5, effort=0.40, risk=0.30,
            dependency_count=0, demand_signal=0.60, urgency=0.20, age=0,
        )
        self.assertAlmostEqual(rice_score(item), 0.96, places=4)

    def test_high_risk_reduces_confidence(self):
        # risk = 0.80 (above 0.7 threshold -> confidence drops to 0.5)
        # RICE = (0.60 * 0.80 * 0.5) / 0.40 = 0.24 / 0.40 = 0.60
        item = BacklogItem(
            item_id=2, name="Risky Item", business_value=0.80,
            effort_raw=5, effort=0.40, risk=0.80,
            dependency_count=0, demand_signal=0.60, urgency=0.20, age=0,
        )
        self.assertAlmostEqual(rice_score(item), 0.60, places=4)


class TestWSJFScore(unittest.TestCase):
    def test_hand_calculated_example(self):
        # Cost of Delay = business_value + urgency + (1 - risk)
        #               = 0.80 + 0.20 + (1 - 0.30) = 0.80 + 0.20 + 0.70 = 1.70
        # WSJF = 1.70 / 0.40 = 4.25
        item = BacklogItem(
            item_id=1, name="Test Item", business_value=0.80,
            effort_raw=5, effort=0.40, risk=0.30,
            dependency_count=0, demand_signal=0.60, urgency=0.20, age=0,
        )
        self.assertAlmostEqual(wsjf_score(item), 4.25, places=4)


class TestAssignTiers(unittest.TestCase):
    def test_capacity_respected(self):
        # 3 items, 5 points each = 15 total. Capacity = 10 -> only 2 fit in "Now".
        items = [
            BacklogItem(item_id=i, name=f"Item {i}", business_value=0.9 - i * 0.1,
                        effort_raw=5, effort=0.4, risk=0.2, dependency_count=0,
                        demand_signal=0.5, urgency=0.3, age=0)
            for i in range(3)
        ]
        tiers, ranked = assign_tiers(items, wsjf_score, capacity_points=10)
        now_count = sum(1 for t in tiers.values() if t == Tier.NOW)
        self.assertEqual(now_count, 2, "Exactly 2 of 3 five-point items should fit in a 10-point capacity")

    def test_highest_score_gets_now_first(self):
        # Item 0 has the highest business_value -> should rank first -> tier NOW
        items = [
            BacklogItem(item_id=i, name=f"Item {i}", business_value=0.9 - i * 0.3,
                        effort_raw=3, effort=0.25, risk=0.2, dependency_count=0,
                        demand_signal=0.5, urgency=0.3, age=0)
            for i in range(3)
        ]
        tiers, ranked = assign_tiers(items, wsjf_score, capacity_points=3)
        self.assertEqual(tiers[0], Tier.NOW)
        self.assertNotEqual(tiers[2], Tier.NOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
