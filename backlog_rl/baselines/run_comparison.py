"""
backlog_rl.baselines.run_comparison
-------------------------------------
Runs RICE, WSJF, and a random policy through IDENTICAL environment
conditions (same seed, same shock sequence) and compares cumulative
reward -- the "common reward, different policies" evaluation design
specified in Chapter 3, Section 3.7.1.

This is a Week 8 baseline-only comparison. The RL policy is added to
this same comparison in Week 9 once trained.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backlog_rl.env.backlog_env import BacklogPrioritisationEnv, MAX_ITEMS
from backlog_rl.baselines.static_scorers import rice_score, wsjf_score, assign_tiers, Tier


def policy_action_from_tiers(env, tiers):
    """Convert an item_id -> Tier mapping into a fixed-size MultiDiscrete action."""
    action = np.zeros(MAX_ITEMS, dtype=np.int64)
    for idx, item in enumerate(env.items[:MAX_ITEMS]):
        action[idx] = int(tiers.get(item.item_id, Tier.LATER))
    return action


def run_episode(policy_name, seed, n_items=30, capacity=20, n_cycles=12):
    env = BacklogPrioritisationEnv(n_items=n_items, capacity_points=capacity, seed=seed)
    obs, info = env.reset(seed=seed)

    rewards = []
    for cycle in range(n_cycles):
        if policy_name == "random":
            action = env.action_space.sample()
        elif policy_name == "rice":
            tiers, _ = assign_tiers(env.items, rice_score, env.capacity_points)
            action = policy_action_from_tiers(env, tiers)
        elif policy_name == "wsjf":
            tiers, _ = assign_tiers(env.items, wsjf_score, env.capacity_points)
            action = policy_action_from_tiers(env, tiers)
        else:
            raise ValueError(policy_name)

        obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards


def main():
    N_SEEDS = 10          # multiple seeds for a fairer comparison
    N_CYCLES = 12

    results = {"random": [], "rice": [], "wsjf": []}

    for seed in range(N_SEEDS):
        for policy in results:
            rewards = run_episode(policy, seed=seed, n_cycles=N_CYCLES)
            results[policy].append(rewards)

    print(f"Baseline comparison over {N_SEEDS} seeds, {N_CYCLES} cycles each\n")
    print(f"{'Policy':10s} {'Mean cumulative reward':>24s} {'Std':>8s}")
    summary = {}
    for policy, episodes in results.items():
        cumulative = [sum(ep) for ep in episodes]
        mean_r = np.mean(cumulative)
        std_r = np.std(cumulative)
        summary[policy] = (mean_r, std_r, cumulative)
        print(f"{policy:10s} {mean_r:24.2f} {std_r:8.2f}")

    # ---- Chart 1: mean cumulative reward bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    policies = list(results.keys())
    means = [summary[p][0] for p in policies]
    stds = [summary[p][1] for p in policies]
    colors = ["#9A8C7A", "#028090", "#0B3D42"]

    axes[0].bar(policies, means, yerr=stds, capsize=6, color=colors)
    axes[0].set_title("Mean Cumulative Reward by Policy\n(10 seeds, 12 cycles each)", fontsize=12)
    axes[0].set_ylabel("Cumulative reward")
    axes[0].axhline(0, color="grey", linewidth=0.8)

    # ---- Chart 2: average reward per cycle (trajectory) ----
    for policy, color in zip(policies, colors):
        avg_per_cycle = np.mean(results[policy], axis=0)
        axes[1].plot(range(1, N_CYCLES + 1), avg_per_cycle, label=policy.upper(), color=color, marker="o", markersize=4)
    axes[1].set_title("Average Reward per Cycle\n(mean across 10 seeds)", fontsize=12)
    axes[1].set_xlabel("Planning cycle")
    axes[1].set_ylabel("Reward")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("/home/claude/impl/week8/baseline_comparison.png", dpi=150)
    print("\nSaved chart: baseline_comparison.png")


if __name__ == "__main__":
    main()
