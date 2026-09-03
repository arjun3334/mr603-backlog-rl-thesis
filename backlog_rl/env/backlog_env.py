"""
backlog_rl.env.backlog_env
----------------------------
Gymnasium-compatible environment implementing the MDP defined in the
MR603 Week 5 MDP Specification and Chapter 3, Sections 3.3 and 3.5.1.

State  : fixed-size (MAX_ITEMS x 7) feature matrix + 3 global features
Action : per-item discrete tier assignment {0: Later, 1: Next, 2: Now}
Reward : Chapter 3 / Week 5 reward function (value - dependency
         violation - capacity overrun - urgency delay)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backlog_rl.env.generator import BacklogGenerator, BacklogItem

MAX_ITEMS = 40
N_FEATURES = 7
EPISODE_LENGTH = 12          # planning cycles per episode (Chapter 3, Section 3.3.3)
SHOCK_PROBABILITY = 0.20     # per-cycle probability of an injected shock

# Default reward weights (Week 5 Specification, Section 3.2)
W1_VALUE = 1.0
W2_DEPENDENCY = 0.5
W3_CAPACITY = 1.0  # CORRECTED (was 0.4) -- closes the reward-exploitation gap diagnosed
                   # in Week 9: old weight allowed a positive net reward from overrunning
                   # capacity with cheap, high-value items. New weight matches W1_VALUE,
                   # so the marginal capacity-overrun penalty per point can never be
                   # smaller than the maximum possible per-unit value gain.
W4_URGENCY = 0.3
URGENCY_THRESHOLD = 0.7


class BacklogPrioritisationEnv(gym.Env):
    """Custom Gymnasium environment for adaptive backlog prioritisation."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_items=30, capacity_points=20, seed=None):
        super().__init__()
        self.n_items_target = n_items
        self.capacity_points = capacity_points
        self.generator = BacklogGenerator(seed=seed)

        # Observation: MAX_ITEMS x N_FEATURES (flattened) + 3 global features
        obs_dim = MAX_ITEMS * N_FEATURES + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Action: one of 3 tiers, per item slot (MultiDiscrete)
        self.action_space = spaces.MultiDiscrete([3] * MAX_ITEMS)

        self.items = []
        self.cycle = 0
        self.last_shock = 0

    # ---------- Gymnasium API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.generator = BacklogGenerator(seed=seed)
        self.items = self.generator.generate_backlog(self.n_items_target)
        self.cycle = 0
        self.last_shock = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        shock_applied = self._maybe_apply_shock()

        reward, info = self._compute_reward_and_advance(action)

        self.cycle += 1
        terminated = False
        truncated = self.cycle >= EPISODE_LENGTH

        info["shock_applied"] = shock_applied
        info["cycle"] = self.cycle
        return self._get_obs(), reward, terminated, truncated, info

    # ---------- internal mechanics ----------

    def _maybe_apply_shock(self):
        if self.generator.rng.random() >= SHOCK_PROBABILITY or not self.items:
            self.last_shock = 0
            return None

        self.last_shock = 1
        shock_type = self.generator.rng.choice(["demand_spike", "dependency_resolved", "capacity_change"])
        target = self.items[self.generator.rng.integers(0, len(self.items))]

        if shock_type == "demand_spike":
            target.demand_signal = min(1.0, target.demand_signal + 0.4)
        elif shock_type == "dependency_resolved":
            target.dependency_count = max(0, target.dependency_count - 1)
        elif shock_type == "capacity_change":
            delta = self.generator.rng.choice([-5, 5])
            self.capacity_points = max(5, self.capacity_points + delta)

        return shock_type

    def _compute_reward_and_advance(self, action):
        value_delivered = 0.0
        dependency_violation = 0
        urgency_delay = 0.0
        now_effort = 0

        now_items, next_items, later_items = [], [], []

        for idx, item in enumerate(self.items):
            tier = int(action[idx]) if idx < len(action) else 0
            if tier == 2:  # Now
                now_items.append(item)
                value_delivered += item.business_value
                now_effort += item.effort_raw
                if item.dependency_count > 0:
                    dependency_violation += 1
            elif tier == 1:
                next_items.append(item)
            else:
                later_items.append(item)

        capacity_overrun = max(0, now_effort - self.capacity_points)

        for item in self.items:
            if item.urgency > URGENCY_THRESHOLD and item not in now_items:
                urgency_delay += item.urgency

        reward = (
            W1_VALUE * value_delivered
            - W2_DEPENDENCY * dependency_violation
            - W3_CAPACITY * capacity_overrun
            - W4_URGENCY * urgency_delay
        )

        # Advance the backlog: completed "Now" items leave, new items enter,
        # everything else ages by one cycle.
        completed_ids = {it.item_id for it in now_items}
        self.items = [it for it in self.items if it.item_id not in completed_ids]
        for it in self.items:
            it.age += 1
        n_new = max(0, self.n_items_target - len(self.items))
        self.items.extend(self.generator.generate_backlog(n_new))

        info = {
            "value_delivered": value_delivered,
            "dependency_violation": dependency_violation,
            "capacity_overrun": capacity_overrun,
            "urgency_delay": urgency_delay,
            "n_now": len(now_items),
            "n_next": len(next_items),
            "n_later": len(later_items),
        }
        return reward, info

    def _get_obs(self):
        mat = np.zeros((MAX_ITEMS, N_FEATURES), dtype=np.float32)
        for i, item in enumerate(self.items[:MAX_ITEMS]):
            mat[i] = item.to_feature_vector()
        global_feats = np.array([
            min(self.capacity_points, 50) / 50.0,
            min(self.cycle, EPISODE_LENGTH) / EPISODE_LENGTH,
            float(self.last_shock),
        ], dtype=np.float32)
        return np.concatenate([mat.flatten(), global_feats])

    def _get_info(self):
        return {"n_active_items": len(self.items), "capacity_points": self.capacity_points}


if __name__ == "__main__":
    env = BacklogPrioritisationEnv(n_items=15, capacity_points=20, seed=7)
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}  (expected {MAX_ITEMS * N_FEATURES + 3})")
    print(f"Active items: {info['n_active_items']}, capacity: {info['capacity_points']}\n")

    total_reward = 0.0
    for cycle in range(EPISODE_LENGTH):
        # Random policy for smoke-testing the environment mechanics
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        shock = f" [SHOCK: {info['shock_applied']}]" if info.get("shock_applied") else ""
        print(f"Cycle {info['cycle']:2d} | reward={reward:6.2f} | value={info['value_delivered']:.2f} "
              f"| dep_viol={info['dependency_violation']} | cap_overrun={info['capacity_overrun']} "
              f"| urg_delay={info['urgency_delay']:.2f} | now/next/later={info['n_now']}/{info['n_next']}/{info['n_later']}{shock}")
        if terminated or truncated:
            break

    print(f"\nTotal episode reward (random policy): {total_reward:.2f}")
