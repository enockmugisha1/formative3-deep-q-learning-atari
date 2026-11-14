# Task 3: Hyperparameter Tuning — Experiment Results

**Member Name:** Jeremie

My experiments use **config set B** in [`experiments.py`](experiments.py). Run them with:

```bash
uv run python experiments.py --member jeremie --set B            # all 10 (long)
uv run python experiments.py --member jeremie --set B --only 3   # a single experiment
uv run python experiments.py --member jeremie --set B --only 1 --steps 15000  # quick smoke test
```

Each run saves a model to `models/jeremie/expNN.pth`, an episode log to
`plots/jeremie/expNN_log.csv`, and a reward curve to `plots/jeremie/expNN_rewards.png`.
Final metrics are collected in `models/jeremie/summary.csv`. All 10 runs used
150,000 environment steps.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1  | 0.00025 | 0.99 | 64 | 1 | 0.05 | 100,000 | Solid baseline. Learned steadily from ~1.3 avg reward (first 50 episodes) to ~14.3 (last 50). Best episode 32, final-20 mean 12.2. |
| 2  | 0.001 | 0.99 | 64 | 1 | 0.05 | 100,000 | Learning rate too high: training was unstable and plateaued early at ~7.9 avg (last 50), barely half the baseline. Best episode only 21. Big gradient steps kept overshooting. |
| 3  | 0.0001 | 0.97 | 64 | 1 | 0.05 | 100,000 | Lower LR + lower gamma learned more slowly; reached ~10.6 avg (last 50), final-20 mean 11.0. Decent but never caught the baseline within 150k steps. |
| 4  | 0.00025 | 0.9 | 64 | 1 | 0.05 | 100,000 | **Worst run.** Gamma 0.9 makes the agent short-sighted (rewards 10+ steps away are nearly ignored), so it undervalues keeping the ball alive. Stalled at ~6.5 avg, final-20 mean 7.6. |
| 5  | 0.00025 | 0.995 | 64 | 1 | 0.05 | 100,000 | High gamma helped: long-term credit for keeping rallies going. Reached ~13.4 avg (last 50), best episode 35, final-20 mean 14.0 — second-best settled score. |
| 6  | 0.00025 | 0.99 | 128 | 1 | 0.05 | 100,000 | Large batch gave smoother, more stable updates: ~14.3 avg (last 50), best episode 41. Slightly slower per step but strong final performance (final-20 mean 13.3). |
| 7  | 0.00025 | 0.99 | 32 | 1 | 0.05 | 100,000 | Small batch was noisier but still fine: ~13.0 avg (last 50), final-20 mean 12.5. Close to baseline, confirming batch size mattered less than LR or gamma here. |
| 8  | 0.00025 | 0.99 | 64 | 1 | 0.05 | 75,000 | **Best run.** Faster epsilon decay meant less time wasted on random actions and more greedy practice within the step budget. Climbed from 0.9 avg to ~15.2 (last 50); best episode 47, final-20 mean 17.1. |
| 9  | 0.00025 | 0.99 | 64 | 1 | 0.02 | 150,000 | Decay too slow: epsilon was still high late in training, so the agent kept acting randomly instead of exploiting. Only ~11.0 avg (last 50), best episode 20. |
| 10 | 0.00025 | 0.99 | 64 | 0.8 | 0.01 | 50,000 | Aggressive exploitation (low start, very fast decay) worked reasonably (~11.6 avg, final-20 mean 13.1) but stayed below experiment 8 — some early exploration is still needed to discover good behaviour. |

## Best and worst configurations

- **Best — Experiment 8** (lr 0.00025, gamma 0.99, batch 64, eps 1.0 → 0.05 over 75k): final-20 mean **17.1**, best episode **47**. With only 150k training steps, decaying epsilon faster gave the agent more time acting greedily on what it had learned.
- **Worst — Experiment 4** (gamma 0.90): final-20 mean **7.6**. In Breakout the payoff for keeping the ball in play arrives many steps later, so a short-sighted discount factor cripples learning. Experiment 2 (lr 0.001) was almost as bad — updates were too large and training never stabilised.

## What each hyperparameter did

- **Learning rate** mattered most: 0.00025 was the sweet spot; 0.001 destabilised training (exp 2) and 0.0001 learned too slowly for the budget (exp 3).
- **Gamma** needs to be high for Breakout: 0.995 ≈ 0.99 > 0.97 > 0.9 (exps 5, 1, 3, 4).
- **Batch size** had a mild effect: 128 was smoothest, 32 noisiest, all workable (exps 6, 7).
- **Epsilon schedule** was the second-biggest lever: decaying to 0.05 over 75k steps beat both slower (exp 9) and more extreme (exp 10) schedules.

## Final configuration (justification)

Experiment 8: **lr 0.00025, gamma 0.99, batch 64, epsilon 1.0 → 0.05 over 75,000 steps.** It combines the stable mid learning rate, a far-sighted discount, and an exploration schedule matched to the 150k-step budget, and it achieved both the highest settled score and the highest single episode of all 10 runs.
