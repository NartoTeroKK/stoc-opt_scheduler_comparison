# `plots.py` — Module Structure

The module is organized into **4 hierarchical plot levels + 2 specialized analytical plots + 1 entry point**. The two `problem_type` values (`convex`, `non-convex`) are treated as independent experimental blocks: no plot ever compares them against each other.

**Primary data source:** `results: dict[str, dict[str, TrainingHistory]]` — the notebook's native data structure. Each `run_name` follows the format `{problem_type}_{scheduler}_{optimizer}_{seed}` and is parsed internally by the module to extract the analysis dimensions. Each `TrainingHistory` instance exposes:
- `history.train_losses_arr` → `np.ndarray` shape `(n_epochs,)`
- `history.train_accuracies_arr` → `np.ndarray` shape `(n_epochs,)`
- `history.learning_rates_arr` → `np.ndarray` shape `(n_epochs,)`
- `history.test_metrics` → `dict` with keys `loss`, `accuracy`

**Seed aggregation:** runs are grouped internally by `(scheduler, optimizer)` and their per-epoch arrays are stacked along an additional axis → `np.ndarray` shape `(5, n_epochs)`. Mean and standard deviation are computed over `axis=0`. A rolling mean with `window=5, center=True, min_periods=1` is applied to the aggregated mean curve before plotting — never to the raw arrays used for quantitative metrics.

---

## Visual Encoding — `SCHEDULER_COLORS` & `OPTIMIZER_STYLES`

All constants are defined at module level and referenced by every plot function. No color or line style is hardcoded locally.

### Scheduler Colors — `SCHEDULER_COLORS: dict[str, str]`

| Scheduler | Color | Hex | Rationale |
|---|---|---|---|
| `none` | Gray | `#7f7f7f` | Neutral baseline — visually recessive |
| `exponential` | Blue | `#1f77b4` | Monotonic decay — cool, stable hue |
| `cosine` | Green | `#2ca02c` | Smooth cyclic schedule — organic hue |
| `cyclic` | Orange | `#ff7f0e` | Oscillating schedule — warm, dynamic hue |
| `one-cycle` | Red | `#d62728` | Aggressive warm-up/decay policy — high-salience hue |

Palette derived from `matplotlib` tab10 — maximum perceptual contrast across 5 values, distinguishable in grayscale and under the most common forms of color vision deficiency (deuteranopia, protanopia).

### Optimizer Line Style — `OPTIMIZER_STYLES: dict[str, dict]`

| Optimizer | Style | `linewidth` | std band `alpha` |
|---|---|---|---|
| `sgd` | Solid `"-"` | `2.0` | `0.15` |
| `adam` | Dashed `"--"` | `2.0` | `0.15` |

The solid/dashed distinction is the standard visual encoding for paired comparisons — it allows simultaneous reading of color (scheduler) and line style (optimizer) without ambiguity, including in black-and-white print.

### Seed Style (Level 3 only) — `SEED_STYLE`

| Element | Style | `linewidth` | `alpha` |
|---|---|---|---|
| Individual run (single seed) | Solid `"-"` | `0.8` | `0.3` |
| Aggregate mean (thick line) | Solid `"-"` | `2.5` | `1.0` |

Individual seed thin lines share the same color as their configuration (from `SCHEDULER_COLORS`) with no style distinction — run-to-run variance is encoded exclusively through alpha and line width, not color.

---

## Level 1 — `plot_global_comparison(results, problem_type, L_star, epsilon, save_path)`

Single figure per problem type, subplot layout `(1, 2)`. **Left subplot — train_loss:** 10 mean±std curves (5 schedulers × 2 optimizers), color-coded by scheduler, line style by optimizer, semitransparent ±1 std band, and an optional **horizontal gray dashed line** at `y = L_star + epsilon` marking the convergence threshold (omitted if parameters are `None`). **Right subplot — learning_rate:** 10 mean curves with the same visual encoding, no std band. Both subplots share `x = epoch`. Data read from `history.train_losses_arr` and `history.learning_rates_arr`. **Purpose:** global overview of the scheduler×optimizer interaction and relative convergence speed analysis.

---

## Level 2 — `plot_by_optimizer(results, problem_type, save_path)`

Single figure with 2 columns (SGD | Adam). Each column contains 5 scheduler curves with mean±std over `history.train_losses_arr`, `y = train_loss`, `x = epoch`, same color palette as Level 1. **Purpose:** isolate the per-scheduler effect for each optimizer independently — makes SGD oscillations and Adam's damped behavior directly comparable across schedulers.

---

## Level 3 — `plot_seed_variance(results, problem_type, save_path)`

Subplot grid `(5 × 2)`. Each cell `(scheduler, optimizer)` shows 5 thin lines at `alpha=0.3` (one per seed, from `history.train_losses_arr`) overlaid with 1 thick mean line at `alpha=1.0`. No rolling mean — raw arrays only. **Purpose:** diagnostic visualization of inter-seed variance and outlier detection. Not included in the main report.

---

## Level 4 — `plot_final_performance(results, problem_type, save_path)`

Single box plot per problem type. `x` axis = 10 configurations `{scheduler}_{optimizer}`, `y` axis = `test_accuracy` from `history.test_metrics["accuracy"]` or final training loss from `history.train_losses_arr[-1]`. Each box represents the distribution over 5 seeds (median, IQR, outliers). **Purpose:** direct numerical comparison of final performance across configurations, suitable for the results section of the report.

---

## Analytical Plot A — `plot_loss_lr_dual(results, problem_type, save_path)`

Subplot grid `(2 × 5)` — 2 rows (SGD | Adam), 5 columns (one per scheduler). Each subplot uses a dual y-axis: **left y-axis** = mean±std `train_loss` from `history.train_losses_arr` (solid line + std band); **right y-axis** = mean `learning_rate` from `history.learning_rates_arr` (dashed line, fixed orange color). `x = epoch`. **Purpose:** expose the direct temporal correlation between learning rate decay and the damping of loss oscillations — core visualization for training stability analysis.

---

## Analytical Plot B — `plot_scheduler_optimizer_heatmap(results, problem_type, metrics_df, save_path)`

Heatmap `(5 × 2)` — rows = schedulers, columns = optimizers. Each cell value = median over 5 seeds of **convergence epoch** (`epochs_to_threshold` from `metrics_df`) or `history.train_losses_arr[-1]`. Diverging colormap centered on the global median, with numeric annotations in each cell. `metrics_df` is pre-computed by the `convergence_metrics` module and passed in by the entry point. **Purpose:** Pareto-optimal comparison of scheduler×optimizer pairs on a single scalar metric, suited for the conclusions section of the report.

---

## Entry Point — `plot_all(results, problem_type, metrics_df, L_star, epsilon, save_dir)`

Receives `results[problem_type]` (the problem-type-specific sub-dict), a pre-computed `metrics_df`, and `save_dir: Path`. Calls all levels and analytical plots in sequence, saving each figure with standardized naming:
```
{problem_type}_level1_global_comparison.png
{problem_type}_level2_by_optimizer.png
{problem_type}_level3_seed_variance.png
{problem_type}_level4_final_performance.png
{problem_type}_analytical_a_loss_lr_dual.png
{problem_type}_analytical_b_heatmap.png
```