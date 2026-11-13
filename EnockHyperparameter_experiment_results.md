# Task 3: Hyperparameter Tuning — Experiment Results

**Member Name:** Enock

My experiments use **config set C** in [`experiments.py`](experiments.py). Run them with:

```bash
uv run python experiments.py --member enock --set C            # all 10 (long)
uv run python experiments.py --member enock --set C --only 3   # a single experiment
uv run python experiments.py --member enock --set C --only 1 --steps 15000  # quick smoke test
```

Each run saves a model to `models/enock/expNN.pth`, an episode log to
`plots/enock/expNN_log.csv`, and a reward curve to `plots/enock/expNN_rewards.png`.
Final metrics are collected in `models/enock/summary.csv`. Fill in the
**Observed behaviour** column from those logs/plots after each run (e.g. best
reward, final 20-episode mean, whether learning was stable or diverged).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1  | 0.00005 | 0.99 | 16 | 1 | 0.05 | 100,000 | _(fill after run)_ |
| 2  | 0.00005 | 0.99 | 128 | 1 | 0.05 | 100,000 | _(fill after run)_ |
| 3  | 0.00005 | 0.9 | 32 | 1 | 0.05 | 100,000 | _(fill after run)_ |
| 4  | 0.00005 | 0.95 | 32 | 1 | 0.05 | 150,000 | _(fill after run)_ |
| 5  | 0.00005 | 0.999 | 32 | 1 | 0.05 | 100,000 | _(fill after run)_ |
| 6  | 0.00005 | 0.99 | 64 | 1 | 0.1 | 200,000 | _(fill after run)_ |
| 7  | 0.00005 | 0.99 | 32 | 1 | 0.02 | 75,000 | _(fill after run)_ |
| 8  | 0.00005 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | _(fill after run)_ |
| 9  | 0.00005 | 0.97 | 32 | 1 | 0.05 | 50,000 | _(fill after run)_ |
| 10 | 0.00005 | 0.995 | 16 | 0.8 | 0.01 | 150,000 | _(fill after run)_ |

## Metrics to report per run (from `summary.csv`)

- `episodes` — number of completed episodes in the run.
- `best_reward` — highest single-episode reward reached.
- `final20_mean_reward` — mean reward over the last 20 episodes (settled performance).
