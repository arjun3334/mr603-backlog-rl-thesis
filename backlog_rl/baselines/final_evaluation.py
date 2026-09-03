"""
backlog_rl.baselines.final_evaluation
----------------------------------------
Addresses the examiner's mandatory Priority 1 items:
  - Held-out TEST seeds, never used during training or reward tuning
  - Multiple independent PPO training seeds (loaded here, not retrained)
  - Paired statistics appropriate for a matched-seed comparison
  - Reports capacity feasibility, dependency violations, urgency delay,
    and adaptation-after-shock -- not just cumulative reward

Uses the CORRECTED reward function (W3_CAPACITY = 1.0).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import PPO

from backlog_rl.env.backlog_env import BacklogPrioritisationEnv
from backlog_rl.baselines.static_scorers import rice_score, wsjf_score, assign_tiers, Tier
from backlog_rl.baselines.run_comparison import policy_action_from_tiers

# Held-out test seeds -- NEVER used during training or reward tuning
TEST_SEEDS = list(range(100, 110))  # 10 held-out seeds
TRAIN_SEEDS = [0, 1, 2]
N_CYCLES = 12
CAPACITY = 20


def run_episode_static(policy_name, seed):
    env = BacklogPrioritisationEnv(n_items=30, capacity_points=CAPACITY, seed=seed)
    obs, info = env.reset(seed=seed)
    metrics = {"reward": [], "capacity_overrun": [], "dependency_violation": [],
               "urgency_delay": [], "shock_cycles": [], "post_shock_reward": []}
    last_shock_cycle = None
    for cycle in range(N_CYCLES):
        if policy_name == "random":
            action = env.action_space.sample()
        elif policy_name == "rice":
            tiers, _ = assign_tiers(env.items, rice_score, env.capacity_points)
            action = policy_action_from_tiers(env, tiers)
        elif policy_name == "wsjf":
            tiers, _ = assign_tiers(env.items, wsjf_score, env.capacity_points)
            action = policy_action_from_tiers(env, tiers)
        obs, reward, terminated, truncated, step_info = env.step(action)
        metrics["reward"].append(reward)
        metrics["capacity_overrun"].append(step_info["capacity_overrun"])
        metrics["dependency_violation"].append(step_info["dependency_violation"])
        metrics["urgency_delay"].append(step_info["urgency_delay"])
        if step_info.get("shock_applied"):
            last_shock_cycle = cycle
        if last_shock_cycle is not None and cycle == last_shock_cycle + 2:
            metrics["post_shock_reward"].append(reward)
        if terminated or truncated:
            break
    return metrics


def run_episode_ppo(model, seed):
    env = BacklogPrioritisationEnv(n_items=30, capacity_points=CAPACITY, seed=seed)
    obs, info = env.reset(seed=seed)
    metrics = {"reward": [], "capacity_overrun": [], "dependency_violation": [],
               "urgency_delay": [], "shock_cycles": [], "post_shock_reward": []}
    last_shock_cycle = None
    for cycle in range(N_CYCLES):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        metrics["reward"].append(reward)
        metrics["capacity_overrun"].append(step_info["capacity_overrun"])
        metrics["dependency_violation"].append(step_info["dependency_violation"])
        metrics["urgency_delay"].append(step_info["urgency_delay"])
        if step_info.get("shock_applied"):
            last_shock_cycle = cycle
        if last_shock_cycle is not None and cycle == last_shock_cycle + 2:
            metrics["post_shock_reward"].append(reward)
        if terminated or truncated:
            break
    return metrics


def summarize(metrics_list):
    cum_rewards = [sum(m["reward"]) for m in metrics_list]
    total_overrun = [sum(m["capacity_overrun"]) for m in metrics_list]
    total_dep_viol = [sum(m["dependency_violation"]) for m in metrics_list]
    total_urgency = [sum(m["urgency_delay"]) for m in metrics_list]
    feasible_cycles = [sum(1 for x in m["capacity_overrun"] if x == 0) for m in metrics_list]
    return {
        "cum_reward": cum_rewards,
        "mean_cum_reward": float(np.mean(cum_rewards)),
        "std_cum_reward": float(np.std(cum_rewards)),
        "mean_capacity_overrun": float(np.mean(total_overrun)),
        "mean_dependency_violation": float(np.mean(total_dep_viol)),
        "mean_urgency_delay": float(np.mean(total_urgency)),
        "pct_feasible_cycles": float(np.mean(feasible_cycles) / N_CYCLES * 100),
    }


def main():
    print("=" * 78)
    print("FINAL EVALUATION -- held-out test seeds, corrected reward, multi-seed PPO")
    print(f"Test seeds (held out, never used in training): {TEST_SEEDS}")
    print(f"PPO training seeds: {TRAIN_SEEDS}")
    print("=" * 78)

    results = {"random": [], "rice": [], "wsjf": []}
    for seed in TEST_SEEDS:
        for policy in ["random", "rice", "wsjf"]:
            results[policy].append(run_episode_static(policy, seed))

    # Load all 3 trained PPO models, evaluate each on all test seeds
    ppo_per_seed_model = {}
    for train_seed in TRAIN_SEEDS:
        model = PPO.load(f"/home/claude/final_push/models/ppo_seed{train_seed}.zip", device="cpu")
        per_test_seed_results = [run_episode_ppo(model, ts) for ts in TEST_SEEDS]
        ppo_per_seed_model[train_seed] = per_test_seed_results

    # PPO "ensemble" per test seed: average cumulative reward across the 3 trained models,
    # for a clean matched-seed pairing against RICE/WSJF on the same test seeds.
    ppo_ensemble_per_testseed = []
    for i, ts in enumerate(TEST_SEEDS):
        cum_rewards_this_testseed = [sum(ppo_per_seed_model[s][i]["reward"]) for s in TRAIN_SEEDS]
        ppo_ensemble_per_testseed.append(np.mean(cum_rewards_this_testseed))

    summary = {p: summarize(results[p]) for p in ["random", "rice", "wsjf"]}
    # PPO summary: pool all 3 seeds x 10 test seeds = 30 episodes for capacity/dependency metrics
    ppo_all_episodes = [ep for s in TRAIN_SEEDS for ep in ppo_per_seed_model[s]]
    summary["ppo"] = summarize(ppo_all_episodes)
    summary["ppo"]["cum_reward"] = ppo_ensemble_per_testseed  # for paired stats (matched to test seeds)
    summary["ppo"]["mean_cum_reward"] = float(np.mean(ppo_ensemble_per_testseed))
    summary["ppo"]["std_cum_reward"] = float(np.std(ppo_ensemble_per_testseed))

    print(f"\n{'Policy':10s} {'Mean CumReward':>15s} {'Std':>8s} {'MeanOverrun':>12s} {'MeanDepViol':>12s} {'%FeasibleCyc':>13s}")
    for p in ["random", "rice", "wsjf", "ppo"]:
        s = summary[p]
        print(f"{p:10s} {s['mean_cum_reward']:15.2f} {s['std_cum_reward']:8.2f} "
              f"{s['mean_capacity_overrun']:12.2f} {s['mean_dependency_violation']:12.2f} {s['pct_feasible_cycles']:13.1f}")

    # Paired statistics: PPO ensemble-mean vs RICE, vs WSJF, on the SAME 10 held-out test seeds
    print("\n--- Paired statistics (matched on held-out test seeds) ---")
    for baseline in ["rice", "wsjf"]:
        baseline_vals = summary[baseline]["cum_reward"]
        ppo_vals = summary["ppo"]["cum_reward"]
        t_stat, p_val = stats.ttest_rel(ppo_vals, baseline_vals)
        diff = np.array(ppo_vals) - np.array(baseline_vals)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else float('nan')
        ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"PPO vs {baseline.upper()}: paired t={t_stat:.3f}, p={p_val:.4f} ({sig}), "
              f"Cohen's d={cohens_d:.3f}, 95% CI of mean diff=[{ci_low:.2f}, {ci_high:.2f}]")

    # Adaptation-after-shock: average reward 2 cycles after a shock, across all policies
    print("\n--- Adaptation after shock (mean reward, 2 cycles post-shock) ---")
    for p in ["random", "rice", "wsjf"]:
        post = [v for m in results[p] for v in m["post_shock_reward"]]
        print(f"{p:10s}: n={len(post)}, mean post-shock reward={np.mean(post) if post else float('nan'):.2f}")
    post_ppo = [v for s in TRAIN_SEEDS for m in ppo_per_seed_model[s] for v in m["post_shock_reward"]]
    print(f"{'ppo':10s}: n={len(post_ppo)}, mean post-shock reward={np.mean(post_ppo) if post_ppo else float('nan'):.2f}")

    # Save everything needed to regenerate figures directly from THIS final output
    np.savez("/home/claude/final_push/final_results.npz",
             test_seeds=TEST_SEEDS, train_seeds=TRAIN_SEEDS,
             random_cum=summary["random"]["cum_reward"], rice_cum=summary["rice"]["cum_reward"],
             wsjf_cum=summary["wsjf"]["cum_reward"], ppo_cum=summary["ppo"]["cum_reward"],
             random_overrun=summary["random"]["mean_capacity_overrun"],
             rice_overrun=summary["rice"]["mean_capacity_overrun"],
             wsjf_overrun=summary["wsjf"]["mean_capacity_overrun"],
             ppo_overrun=summary["ppo"]["mean_capacity_overrun"])
    print("\nSaved final_results.npz for figure regeneration.")


if __name__ == "__main__":
    main()
