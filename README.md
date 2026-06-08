# Stochastic Optimization: A Comparison of Learning Rate Scheduling Strategies

**Course project for *Metodi di Ottimizzazione Stocastici* (MSc in Artificial Intelligence, Big Data & Data Science)**  
Università degli Studi di Ferrara · Author: **Marco Perozzi**

---

## Overview

This project systematically compares **learning rate scheduling strategies** for **stochastic optimization** in neural network training. The study evaluates five schedulers across two optimizers on both convex and non-convex problems, using a rigorous repeated-seed experimental design with quantitative convergence metrics.

### Schedulers compared

| Scheduler | Type | Behavior |
|-----------|------|----------|
| **None** (constant LR) | Baseline | Fixed learning rate |
| **Exponential Decay** | Monotonic decay | `γ`-factor reduction per epoch |
| **Cosine Annealing** | Smooth decay | Cosine-shaped LR cycle |
| **Cyclical LR** (CLR) | Cyclic | Triangular2 oscillation between bounds |
| **One-Cycle** | Cyclic | Single cycle: warmup → annealing |

### Optimizers

- **SGD** (Stochastic Gradient Descent, lr = 0.01)
- **Adam** (Adaptive Moment Estimation, lr = 0.001)

### Problems

| Problem type | Model | Dataset | Task |
|-------------|-------|---------|------|
| **Convex** | Logistic Regression | Breast Cancer (Wisconsin) | Binary classification, 30 features |
| **Non-convex** | MLP (2 hidden layers, 256→128) | MNIST | 10-class digit classification, 784 features |

---

## Metrics

All metrics are computed **per seed** and then **aggregated** (mean ± std) across 10 random seeds:

| Metric | Definition | What it measures |
|--------|-----------|-----------------|
| **EtT** (Epochs to Target) | First epoch s.t. `L_t ≤ L* + 0.05·(L₀ − L*)` | Convergence speed |
| **AUL** (Area Under Loss) | `∫ L_t dt` (trapezoidal) | Cumulative loss (lower = faster drop) |
| **Suboptimality gap** | `L_T − L*` | Final solution quality |
| **CV** (Coefficient of Variation) | `σ/μ` over loss curve (post warm-up) | Training stability |
| **Gradient norm** | Per-epoch ℓ₂ gradient norm | Optimization landscape smoothness |
| **Convergence rate** | Proportion of seeds reaching L_target | Reliability |

**L*** is the reference optimal loss calculated as the empirical minimum from an extended training run of the best scheduler–optimizer configuration.

---

## Results

Pre-computed results are available in:

| File | Contents |
|------|----------|
| `results/results_v6.pkl` | Full `TrainingHistory` for all 200 runs |
| `results/convex_metrics.csv` | Aggregated metrics (convex) |
| `results/non-convex_metrics.csv` | Aggregated metrics (non-convex) |
| `reports/figures/` | 20 publication-ready figures |
| `reports/report.pdf` | Final written report |

The MLflow experiment database is stored in `mlruns/`:
```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

---

## Project Structure

```
├── configs/
│   └── experiment_config.yaml     # Hyperparameters & experiment matrix
├── notebooks/
│   └── experiment.ipynb           # Single notebook orchestrating all experiments
├── src/stoc_opt_scheduler_comparison/
│   ├── config.py                  # Typed config dataclass + YAML loader
│   ├── utils.py                   # Logging utilities, inference_mode decorator
│   ├── data_load/
│   │   ├── loaders.py             # Dataset factory (synthetic, breast_cancer, mnist)
│   │   └── transforms.py          # Normalization (standard, MNIST canonical)
│   ├── models/
│   │   └── architectures.py       # LogisticRegression, MLP + model factory
│   ├── training/
│   │   ├── engine.py              # train_one_epoch(), evaluate(), train_loop()
│   │   ├── optimizers.py          # SGD/Adam factory
│   │   ├── schedulers.py          # LR scheduler factory + dynamic parameter computation
│   │   └── lstar.py               # L* computation (convex: L-BFGS, non-convex: extended run)
│   ├── evaluation/
│   │   ├── metrics.py             # TrainingHistory, accuracy, confusion matrix, stability
│   │   └── convergence.py         # EtT, AUL, CV, suboptimality gap, gradient norm stats
│   ├── tracking/
│   │   └── mlflow_manager.py      # MLflow experiment logging & result reconstruction
│   └── visualization/
│       └── plots.py               # 4-level plot hierarchy + analytical plots
├── results/                       # Pickled results, CSV metrics
├── reports/figures/               # Generated figures (20 files)
├── data/                          # Dataset storage (auto-downloaded)
├── pyproject.toml                 # Project metadata, dependencies, build config
└── README.md (this file)
```

### Visualization hierarchy

All plots follow a consistent visual encoding (scheduler → color, optimizer → line style):

1. **Level 1 — Global Comparison**: 3×1 subplots (loss linear, loss log, LR schedules)
2. **Level 2 — By Optimizer**: 2×2 grid (SGD/Adam rows, linear/log columns) with mean±std bands
3. **Level 3 — Seed Variance**: 5×2 grid of individual seed trajectories + mean overlay
4. **Level 4 — Final Performance**: Dot plots with error bars (train last/min, test)
- **Analytical A**: Loss + LR dual-axis per config
- **Analytical B**: Heatmaps of scalar metrics across scheduler×optimizer
- **Convergence boxplot**: EtT distribution with timeout markers

---

## Reproduction

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (optional, falls back to CPU)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

The project uses a **src-layout** package: all reusable logic lives in `src/stoc_opt_scheduler_comparison/`, which is installed as an editable local package. The notebook in `notebooks/` imports from it via standard `from stoc_opt_scheduler_comparison import ...` statements.

Two equivalent installation methods:

#### Option A — `uv sync` (recommended)

```bash
git clone <repo-url>
cd stoc-opt_scheduler_comparison

# Creates .venv, installs all dependencies, installs the local package in editable mode
uv sync

# Activate the environment (if not automatic)
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` for logical dependencies and `uv.lock` for exact version pinning. The local package is installed automatically via `[tool.uv.sources]` — no extra steps needed.

#### Option B — `pip`

```bash
git clone <repo-url>
cd stoc-opt_scheduler_comparison

python -m venv .venv
source .venv/bin/activate

# Installs the local package + all its dependencies in editable mode
pip install -e .
```

Both methods produce the same result: the package `stoc_opt_scheduler_comparison` is on `sys.path` and the notebook can import it directly.

### Run experiments

```bash
# Launch the complete experiment notebook
uv run jupyter notebook notebooks/experiment.ipynb

# Or run specific sections via the notebook UI
```

The notebook is self-contained:
1. Loads configuration from `configs/experiment_config.yaml`
2. Runs all 200 (2 problems × 5 schedulers × 2 optimizers × 10 seeds) experiments
3. Logs to MLflow
4. Computes convergence metrics
5. Generates all figures
6. Saves results to `results/`

### View experiment tracking

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{perozzi_scheduler_comparison,
  author = {Marco Perozzi},
  title = {Confronto tra Strategie di Scheduling del Learning Rate e Interazione con gli Ottimizzatori},
  year = {2026},
  school = {Universit\`a degli Studi di Ferrara},
  note = {Course project for Metodi di Ottimizzazione Stocastici}
}
```
