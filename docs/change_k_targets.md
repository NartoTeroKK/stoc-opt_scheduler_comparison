# Modify some implementation logic points

## Differents epochs form two problems

In the original experiment implementation, we have two problems: the convex one and the non-convex and the epochs are fixed to 100 for both problems.
Modify:
- For the convex problem: 50 epochs
- For the non-convex problem: 100 epochs

This change must be integrated in the config file `experiment_config.yaml` and it must be propagated in the `config` module and in the `experiment.ipynb` notebook.

## Optimal Loss Estimation, E_target and Sub-optimality Gap Computation, and Convergence Velocity Metric Visualization

To conduct a rigorous convergence velocity analysis based on the sub-optimality gap, establishing a reference value for the optimal loss is mandatory for both experimental tasks. Due to the fundamentally different geometric nature of the underlying loss landscapes, two distinct and standardized experimental protocols must be adopted.

### 1. Protocol for the Convex Task: Logistic Regression on Breast Cancer

Within the context of convex optimization, the cost function exhibits a single, unique analytical global minimum. To identify this exact reference value, mathematically denoted as `L_star_convex`, the following guidelines apply:

*   **Utilization of Deterministic Second-Order Solvers:** The L-BFGS algorithm must be applied to the entire Breast Cancer training dataset (full-batch optimization).
*   **Value Extraction:** The final loss computed by this deterministic process establishes the absolute value of L*, which serves as the lower asymptote for all subsequent convergence velocity evaluations.

Develop dedicated functions to automate this global minimum search process and to compute the E_target and sub-optimality gap metrics during the model training phases. Import the logistic regression model class and the Breast Cancer dataset loader from their respective specific modules, and create a wrapper function within the training module that executes the L-BFGS optimization and returns the L* value.

Use the following code snippet as a structural guideline to implement the global minimum search function:

```python
clf = LogisticRegression(
    solver='lbfgs', 
    penalty=None, 
    max_iter=10000, 
    tol=1e-12 
)

clf.fit(...)
y_pred_proba = clf.predict_proba(...)
L_star = log_loss(y_true, y_pred_proba)

```

For metric calculation, implement dedicated functions within the `convergence.py` module. Given the loss values recorded during training and the reference value L*, these functions must return E_target (the number of epochs required to reach a specific tolerance threshold) and the sub-optimality gap (the difference between the final achieved loss and L*). 

For the convex problem, three distinct E_target metrics must be calculated, corresponding to three different tolerance levels: `1e-2`, `1e-3`, and `1e-4`, respectively named `E_target_lv1`, `E_target_lv2`, and `E_target_lv3`. These threshold levels must also be visualized by integrating them using standard conventions into the pre-existing plotting functions: `plot_by_optimizer()` and `plot_seed_variance()`.


### 2. Protocol for the Non-Convex Task: Multilayer Perceptron (MLP) on MNIST

In deep neural networks, the cost surface is highly non-convex, making the determination of the absolute global minimum theoretically and computationally impossible. Consequently, a Reference Empirical Minimum, denoted as `L_star_non-convex`, is defined by following this precise sequence of guidelines:

*   **Selection of the Top Configuration:** The optimizer and scheduler combination that demonstrated the greatest descent capability during the initial exploration phase—specifically, the one that achieved the lowest training loss value at the end of the standard 100 epochs—is identified.
*   **Execution of an Extended Training Run:** An optimization process is launched for twice the number of epochs of the standard experiment, setting the target to 200 total epochs.
*   **Enforced Learning Rate Decay:** During the second half of this extended run, a drastic and progressive reduction of the learning rate is enforced to allow the model parameters to stabilize at the bottom of the intercepted local minimum.
*   **Value Extraction:** The lowest Cross-Entropy value recorded on the training set at the end of the 200 epochs is formally adopted as `L_star_non-convex`. This value replaces the global minimum for computing the convergence gap in the non-convex task analysis.

In this case as well, it is necessary to implement a dedicated function within the training module that executes this procedure and returns the `L_star_non-convex` reference value, reusing the existing functions from all the involved modules (dataset, model, training loop).

Regarding the computation of the E_target and sub-optimality gap metrics, these must be calculated differently given the distinct nature of the problem and the reference minimum search. For this task, three more permissive tolerance levels will be adopted, set at 1%, 2.5%, and 5% above the `L_star_non-convex` value, and named `E_target_lv1`, `E_target_lv2`, and `E_target_lv3` respectively. These tolerance levels must be clearly and distinctly visualized within the existing plotting functions, using graphical conventions that highlight their relative and more permissive nature compared to the convex task.

## Target Loss: Tolerance for Convergence Analysis

To complete the convergence analysis, it is necessary to define a specific target loss, denoted as `E_target`, which represents the number of epochs required for the loss curve to reach a value below a pre-established tolerance threshold relative to the reference minimum (L*).

The target loss must be calculated for each optimizer and scheduler combination, and it must be clearly visualized within the existing plotting functions, using graphical conventions that emphasize its role as a critical threshold for convergence velocity analysis.

### 1. Technical Instructions: Global Initial Loss Computation

To ensure a scientifically rigorous evaluation, the threshold used to determine the convergence epoch must be identical across all configurations. Calculating a per-configuration initial loss would result in moving target thresholds, invalidating the comparative analysis. The standard procedure requires computing a single **Global Initial Loss** ($L_0$) as the empirical mean across all combinations and seeds within the same problem.
This value serves as the baseline for defining the **Global Target Loss** ($L_{target}$) using the formula:
$$
L_{target} = L^* + \epsilon \cdot (L_0 - L^*)
$$
where $\epsilon$ is the relative tolerance level diffent for the convex and non-convex problems


## 2. Structural Design of the Convergence Boxplot

The boxplot visualizes the **Epochs-to-Target** distribution, providing an engineering metric for computational efficiency and algorithmic stability under stochastic noise.

### Data Requirements:
For each configuration (scheduler x optimizer), you must extract a 1D vector containing exactly **10 integer values**. Each integer represents the first epoch index where a specific seed fell below $L_{target}$ (for loss) or crossed $A_{target}$ (for accuracy).
* If a seed successfully converges at Epoch 14, its value is `14`.
* If a seed fails to reach the threshold within the training budget, it is assigned a timeout value equal to the maximum number of epochs for that problem.

### Visual Mapping:
* **X-Axis:** Discrete categorical labels representing the unique configurations  of the experiment.
* **Y-Axis:** Continuous numerical scale representing the number of epochs to reach the target ($K_{target}$), bounded from `-5` to `105`.
* **The Box:** Represents the Interquartile Range (IQR, middle 50% of the seeds). The horizontal line inside the box marks the **Median** convergence speed.
* **The Whiskers & Fliers:** Extend to show the total spread and variance of the seeds. 
* **The Overlay (Stripplot):** Plots all 10 raw data points as individual markers directly on top of the boxes to explicitly visualize the distribution of every single seed.

Consider this code snippet as a structural guideline for implementing the convergence velocity boxplot visualization function, which should be integrated into the existing plotting module and the exinsting data managemt strategies for the experiment, ensuring that the necessary data is correctly extracted and processed to fit the required input format for the visualization:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def analyze_and_plot_convergence(
    results: Dict[str, np.ndarray], 
    f_x_star: float, 
    epsilon: float = 0.05, 
    max_epochs: int = 100
) -> None:
    """
    Computes the global baseline target, extracts convergence epochs, 
    and generates a standardized publication-ready boxplot.
    
    Args:
        results: Dictionary mapping configuration names to numpy arrays of shape (10, 100).
        f_x_star: The optimal global minimum loss value.
        epsilon: Relative tolerance threshold percentage (e.g., 0.05 for 5%).
        max_epochs: Maximum epoch limit of the training loop.
    """
    # Step 1: Compute the Global Initial Loss (L0) across ALL configurations and seeds at Epoch 0
    epoch_zero_losses = [config_matrix[:, 0] for config_matrix in results.values()]
    global_initial_loss = float(np.mean(epoch_zero_losses))
    
    # Step 2: Compute the rigorous uniform target threshold
    target_loss = f_x_star + epsilon * (global_initial_loss - f_x_star)
    
    print(f"=== Convergence Parameters ===")
    print(f"Global Initial Loss (L0): {global_initial_loss:.6f}")
    print(f"Global Target Loss (L_target): {target_loss:.6f}\n")
    
    # Step 3: Vectorized extraction of the convergence epoch (K_target) for each seed
    processed_records = []
    
    for config_name, loss_matrix in results.items():
        # Mask where condition is met (True if under or equal to target)
        condition_mask = (loss_matrix <= target_loss)
        
        # Extract index of first True value along the epoch axis
        first_hit_epochs = np.argmax(condition_mask, axis=1)
        
        # Check for seeds that never hit the target condition
        never_converged = ~condition_mask.any(axis=1)
        
        # Replace 0-index false positives with max_epochs for non-converged runs
        final_k_targets = np.where(never_converged, max_epochs, first_hit_epochs)
        
        for seed_idx, k_target in enumerate(final_k_targets):
            processed_records.append({
                'Configuration': config_name,
                'Seed': seed_idx,
                'Epochs_to_Target': k_target
            })
            
    df_convergence = pd.DataFrame(processed_records)
    
    # Step 4: Construct the standardized statistical visualization
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Render underlying distribution metrics via Boxplot
    sns.boxplot(
        data=df_convergence, 
        x='Configuration', 
        y='Epochs_to_Target', 
        hue='Configuration',
        palette="viridis", 
        width=0.45,
        linewidth=1.5,
        showfliers=False,
        legend=False
    )
    
    # Superimpose actual seed observations via Stripplot
    sns.stripplot(
        data=df_convergence, 
        x='Configuration', 
        y='Epochs_to_Target', 
        color="black", 
        alpha=0.6, 
        size=6, 
        jitter=0.15
    )
    
    # Layout and labels adjustments
    plt.title("Convergence Velocity Analysis (Time-to-Target Summary)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Optimizer + Scheduler Combinations", fontsize=14, labelpad=15)
    plt.ylabel(f"Epochs to Reach Suboptimality Target ({target_loss:.4f})", fontsize=14, labelpad=15)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(-5, max_epochs + 5)
    
    # Visual anchor for timeout/non-convergence
    plt.axhline(y=max_epochs, color='crimson', linestyle='--', alpha=0.4, label='Training Limit (Max Epochs)')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
```


