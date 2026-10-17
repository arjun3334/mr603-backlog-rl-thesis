# Revision Protocol (fixed before running the revision experiments)

**Study:** Explainable reinforcement learning for product backlog prioritisation (MR603)
**Author:** Bhimarjun Subedi (MIT250661)
**Purpose:** Address the reviewer's critical points C1 to C5 and M3 to M5 on the manuscript.
**Status:** This file is committed to Git **before** `Revision_Experiments.ipynb` is run. The commit timestamp is the evidence that the protocol came first.

## 1. Reason for the revision
After the frozen run (commit df85c258), the environment code was found to keep the team capacity across training episodes. `reset()` did not restore `capacity_points`, so capacity shocks (±5 points) accumulated over about 8,192 training episodes. Test episodes always start at 20 points. Training and test conditions therefore differed.

## 2. Fixed settings (unchanged from the frozen protocol)
- Environment: 30 active items, 7 features, 40 input slots, 12 sprints per episode, shock probability 0.20, starting capacity 20 story points.
- Algorithm: PPO (Stable-Baselines3), MLP policy and value networks of 2 × 128 units, learning rate linear from 3e-4 to 3e-5, n_steps 2048, batch size 64, gamma 0.99, clip range 0.2, other settings at library defaults.
- Training: 98,304 steps per seed; training seeds 10 to 17 (8 seeds).
- Test: scenarios 200 to 214 (15 scenarios). Test reward always uses the corrected weights (w1 = 1.0, w2 = 0.5, w3 = 1.0, w4 = 0.3).
- Unit of analysis: the mean test reward of each training seed over the 15 scenarios (8 values per condition).

## 3. The only code change
`FixedBacklogEnv.reset()` restores `capacity_points` to its starting value (20) at the start of every episode. Nothing else changes.

## 4. Experiments
**Part A (frozen models, no training):**
1. Corrected statistics: comparison with the baseline **means** (RICE, WSJF), Holm step-down correction, bootstrap 95% CI (10,000 resamples, seed 42) of the median difference, Spearman and leave-one-seed-out correlations between training and test reward.
2. Decoding diagnostics: items in Now per sprint and capacity overrun for deterministic, sampled and capacity-aware decoding.
3. Controls with the same capacity fill: trained-network ranking, random ranking, untrained-network ranking, value-only, value per effort, smallest effort first, immediate-gain per effort, and a per-sprint (myopic) knapsack optimum.
4. Shocks on (0.20) versus off (0.0), and reward in sprints with versus without a shock.

**Part B (retraining):** 8 seeds with the fixed environment and w3 = 1.0. If time allows, 8 more seeds with w3 = 0.4 (factorial of training weight × decoding rule).

**Part C (evaluation of retrained models):** deterministic, sampled and capacity-aware decoding on the 15 test scenarios.

## 5. Hypotheses and analysis (fixed in advance)
- **H1 (bug effect):** With the fix, deterministic decoding places fewer items in Now and has a smaller capacity overrun than the frozen models.
- **H2 (performance):** Primary outcome: deterministic decoding of the retrained w3 = 1.0 policies. Two-sided Wilcoxon signed-rank test of the 8 seed means against each baseline mean, Holm-corrected across the two baselines, alpha 0.05.
- **Equivalence:** a 95% bootstrap CI fully inside ±5.0 (the original margin, kept for comparability). The narrower ±2.2 (half the RICE to WSJF gap) is also reported as a sensitivity check.
- **H3 (training signal):** The training-versus-test correlation is reported (Pearson and Spearman) with no direction claimed in advance.
- **Controls:** the trained ranking is considered to add value only if it beats random ranking with the same capacity fill (paired Wilcoxon over the 8 seeds, p < 0.05).

## 6. Reporting rules
- Every seed is reported. No seed is dropped or re-run.
- Results are reported whatever their direction, including a null or negative result.
- Any analysis not listed above is labelled **exploratory** in the paper.
- The frozen results (commit df85c258) remain the original record and are not overwritten.
