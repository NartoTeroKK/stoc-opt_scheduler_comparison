You need to revise the convergence metric computation logic to comply with academic and scientific standards.

1. **Convergence metrics on aggregated loss curves**  
   - Compute all convergence metrics on the *average loss curve* for each configuration, instead of computing metrics on individual run curves and aggregating them afterward.  
   - For each configuration, first aggregate all runs by computing the mean loss at each epoch across runs (i.e., obtain a single average loss trajectory per configuration).  
   - Apply all convergence metrics (sub-optimality gap, AUL norm, E_target, etc.) directly on these average loss curves.

2. **Updated experiment logic and aggregation**  
   - Modify the experiment logic so that, for each configuration, all runs are first aggregated to obtain the average loss curve.  
   - After this aggregation step, perform the convergence metric calculations on the aggregated curve only.  
   - Preserve the existing pre-aggregation metric logic for individual runs, as it is required for boxplots and for computing the number of converging runs over the total.

3. **Replacement of EtT with E_target**  
   - Replace the current EtT (Epochs to Threshold) metric with the new E_target definition.  
   - E_target is defined as the deterministic number of epochs required to reach a target loss level L_target on the *average loss curve* for a configuration.  
   - To determine a single value of E_target per configuration, select L_target using the highest tolerance level, i.e. `lv1 = 0.05`.  
   - Use this L_target consistently across configurations when computing E_target.

4. **Preservation of pre-aggregation metrics**  
   - Do **not** modify the pre-aggregation computation on individual runs, as it is needed for:  
     - Boxplot visualizations of run-level metrics.  
     - Counting the number of converging runs over the total, which must be included as a final metric in the experiment summary table used for convergence and best-configuration analysis.

5. **Notebook tables for analysis**  
   In the notebook, add clearly structured tables to support three distinct analytical views:

   - **Convergence analysis**  
     - Table with records sorted by appropriate convergence criteria.  
     - Metrics to include:  
       - Sub-optimality gap  
       - AUL norm  
       - Number of converging runs (`n_converging_runs`)  
       - E_target  

   - **Stability analysis**  
     - Table focused on stability-related metrics.  
     - Metrics to include:  
       - CV_final (coefficient of variation on final loss)  
       - SI_asymptotic (asymptotic stability index)  
       - RV (relative variability or analogous stability indicator used in the experiment)

   - **Results analysis**  
     - Table summarizing final performance metrics per configuration.  
     - Metrics to include:  
       - Final training loss  
       - Final test loss  
       - Training accuracy  
       - Test accuracy  

Some implementation is being manually modified by me based on well-founded reasons and because I think it is more correct and standard. Please note these changes and do not revert what I changed from the previous agent edit run.
Ensure that naming conventions for metrics and columns are consistent across all tables and that the new logic is fully integrated into the experiment pipeline before generating the tables.
Ensure that all changes are thoroughly tested to confirm that the new convergence metrics are computed correctly on the average loss curves and that the E_target metric is accurately calculated based on the defined L_target.
Ensure that the final tables in the notebook are well-formatted and clearly present the new metrics for analysis.
Ensure that only what is described in the instructions is changed, and that all other aspects of the experiment logic, metric computations, and visualizations remain intact.