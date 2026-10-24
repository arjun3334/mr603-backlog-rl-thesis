# MR603: Evaluating Reinforcement Learning for Product Backlog Prioritisation

Code and evidence for the paper *"Evaluating Reinforcement Learning for Product Backlog Prioritisation: Why Training Reward Misled"* (B. Subedi, M. Awadallah, A. Sharma, Melbourne Institute of Technology).

## Commits referenced in the paper

| Phase | Commit | Contents |
|---|---|---|
| Frozen protocol | `df85c258a511eed9bd6744ffafc07ff0c76229a2` | `Final-implementation.ipynb`, 8 trained models (seeds 10 to 17), `frozen_protocol_results.npz` |
| Follow-up (revision) protocol | `ff048c5b44af2d93f579fd272c8cd3c608b2e046` | `REVISION_PROTOCOL.md` and the revision notebook, committed before the revision runs |
| Revision results | latest commit on branch `revision-oct2026` | Revision notebook outputs, retrained models, `revision_partA.json`, `revision_partC.json` |

## Environment

Windows 11, Intel Core i5-1135G7 (CPU only), Python with Gymnasium 1.3.0, NumPy 2.1.3, SciPy 1.15.3, PyTorch 2.13.0 and Stable-Baselines3 2.9.0. Training takes about 21 minutes per seed on this machine.

## Files

| File | Purpose |
|---|---|
| `Final-implementation.ipynb` | Frozen protocol: environment, RICE/WSJF baselines, PPO training (seeds 10 to 17), evaluation, capacity-aware decoding, explanation layer |
| `models/ppo_seed10..17.zip` | Frozen-protocol policies (original environment) |
| `models_fixed/` | Policies retrained in the corrected environment, w3 = 1.0 |
| `models_fixed_w3_04/` | Policies retrained in the corrected environment, w3 = 0.4 |
| `REVISION_PROTOCOL.md` | Follow-up protocol, committed before the follow-up runs |
| `Revision_Experiments_v2.ipynb` | Follow-up experiments (Parts A, B and C below) |
| `revision_partA.json` | Corrected statistics, decoding diagnostics, ranking controls, shock analysis and capacity-drift reconstruction |
| `revision_partC.json` | Evaluation of the retrained policies |

## Reproducing the results

Open `Revision_Experiments_v2.ipynb` in Jupyter and run the cells in order. Cell 0 sets the working folder; edit the path there if you clone elsewhere.

- **Part A (Cells 0 to 8, about 15 minutes, no training).** Reloads the frozen models. Cell 3 must reproduce the frozen results exactly (deterministic mean −1300.49, capacity-aware mean 12.21). Produces the statistics in Tables IX and X, the controls in Table XI, the shock comparison in Table XII and the capacity-drift reconstruction in Section IV-E. Writes `revision_partA.json`.
- **Part B (about 4 hours).** Checks that the corrected environment restores capacity to 20 story points at the start of every episode, then retrains seeds 10 to 17 with w3 = 1.0 and w3 = 0.4. Interrupted runs resume where they stopped.
- **Part C (a few minutes).** Evaluates the retrained policies under the common test reward (w3 = 1.0) and writes `revision_partC.json` (Table XIII).

Test scenarios use seeds 200 to 214. Deterministic and capacity-aware results reproduce exactly. Sampled-decoding and random-baseline values can vary slightly because those evaluations are not seeded.

## Known defect in the frozen run

In the original environment, `reset()` did not restore team capacity, so capacity shocks accumulated across training episodes. The frozen results are kept as recorded. `FixedBacklogEnv` in the revision notebook restores capacity on reset, and the retrained results use it. See Sections IV-E and IV-H of the paper.
