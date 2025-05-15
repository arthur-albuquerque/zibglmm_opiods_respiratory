# ZIBGLMM for Meta-Analysis of Opioid-Related Respiratory Outcomes

## Overview

This repository implements a Zero-Inflated Bivariate Generalized Linear Mixed Model (ZIBGLMM) for meta-analysis of opioid-related respiratory outcomes, as described in:

> "ZIBGLMM: Zero-Inflated Bivariate Generalized Linear Mixed Model for Meta-Analysis with Double-Zero-Event Studies"  
> [DOI: 10.1017/rsm.2024.4](https://www.cambridge.org/core/journals/research-synthesis-methods/article/zibglmm-zeroinflated-bivariate-generalized-linear-mixed-model-for-metaanalysis-with-doublezeroevent-studies/FCDCE1CC52319606DE9294F776411A3E)

The model is adapted to include meta-regression for subgroup analysis, applied to a dataset of 84 studies on respiratory outcomes associated with opioid use. The repository contains data, Stan models, and Quarto files with R code for prior predictive checks, model diagnostics, posterior analysis, and visualization, handling double-zero-event studies (where both treatment and control groups report zero events).

## Repository Structure

The repository is organized as follows:

```
zibglmm_opiods_respiratory/
│
├── analyses/
│   ├── 01_prior_predictive.qmd           # Prior predictive checks for overall model
│   ├── 02_posterior_diagnostics_ppc.qmd  # Diagnostics and posterior predictive checks for overall model
│   ├── 03_posterior_model_comparison.qmd # Comparison of overall models with different priors
│   ├── 04_results.qmd                    # Summary figures for overall and subgroups models
│   └── subgroups/
│       ├── 01_prior_predictive.qmd       # Prior predictive checks for subgroups model
│       ├── 02_posterior_diagnostics_ppc.qmd # Diagnostics and posterior predictive checks for subgroups model
│       ├── 03_posterior_model_comparison.qmd # Comparison of subgroups models with different priors
│
├── data/
│   └── raw_data.xlsx                     # Dataset with 84 studies
│
├── models/
│   ├── stan/
│   │   ├── zibglmm_model1.stan           # Overall ZIBGLMM model (prior set 1)
│   │   ├── zibglmm_model2.stan           # Overall ZIBGLMM model (prior set 2)
│   │   ├── zibglmm_prior_model1.stan     # Prior predictive check for model 1
│   │   ├── zibglmm_prior_model2.stan     # Prior predictive check for model 2
│   │   ├── zibglmm_subgroups_model1.stan # Subgroups ZIBGLMM model (prior set 1)
│   │   ├── zibglmm_subgroups_model2.stan # Subgroups ZIBGLMM model (prior set 2)
│   │   ├── zibglmm_subgroups_prior_model1.stan # Prior predictive check for subgroups model 1
│   │   ├── zibglmm_subgroups_prior_model2.stan # Prior predictive check for subgroups model 2
│   │   ├── original_zibglmm_model.stan # Original article model, for reference
│   └── storage/
│       └── [model files]                 # Files for loading fitted models (cmdstanr)
│
└── README.md                             # This file
```

## Prerequisites

To run the analyses, install the following:

- **R** (version 4.2.0 or higher)
- **RStudio** (recommended for Quarto files)
- **CmdStanR**

Install CmdStan: [cmdstanr documentation](https://mc-stan.org/cmdstanr/).

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arthur-albuquerque/zibglmm_opiods_respiratory.git
   cd zibglmm_opiods_respiratory
   ```

2. **Prepare the Environment**:
   - Configure CmdStan with `cmdstanr`.
   - Ensure `raw_data.xlsx` in `data/` is accessible.

3. **Run the Analyses**:
   Render Quarto files in `analyses/` using RStudio or Quarto CLI:

   - **Overall Model**:
     ```bash
     quarto render analyses/01_prior_predictive.qmd
     quarto render analyses/02_posterior_diagnostics_ppc.qmd
     quarto render analyses/03_posterior_model_comparison.qmd
     quarto render analyses/04_results.qmd
     ```

   - **Subgroups Model**:
     ```bash
     quarto render analyses/subgroups/01_prior_predictive.qmd
     quarto render analyses/subgroups/02_posterior_diagnostics_ppc.qmd
     quarto render analyses/subgroups/03_posterior_model_comparison.qmd
     ```

   Alternatively, use RStudio’s "Render" button for each `.qmd` file.

4. **View Results**:
   - Outputs (figures, tables, diagnostics) are saved in `analyses/` or its subfolders.
   - `04_results.qmd` generates a summarizing figure comparing overall and subgroups models.
   - Fitted models in `models/storage/` can be loaded with `cmdstanr` for further analysis.

## Methodology

The ZIBGLMM models double-zero-event studies in meta-analysis, addressing excess zeros in respiratory outcome data (e.g., no events in both treatment and control groups). Key features include:
- **Zero-Inflation**: A Bernoulli component models the probability of zero events due to structural zeros.
- **Bivariate Structure**: Jointly models treatment and control outcomes, accounting for correlation.
- **Mixed Effects**: Random effects capture study-level heterogeneity.
- **Meta-Regression**: Subgroups models incorporate covariates to explore heterogeneity across subgroups.

Two model variants are provided for both overall and subgroups analyses:
- **Overall Models** (`zibglmm_model1.stan`, `zibglmm_model2.stan`): Estimate overall opioid effects, differing in prior specifications.
- **Subgroups Models** (`zibglmm_subgroups_model1.stan`, `zibglmm_subgroups_model2.stan`): Include meta-regression covariates for subgroup effects.

Bayesian inference is implemented using Stan via `cmdstanr`, with prior predictive checks, posterior diagnostics, and model comparisons to ensure robustness.

## Model Modifications

The models in this repository (e.g., `zibglmm_model1.stan`) are modified from the original model (`original_zibglmm_model.stan`) in the article to improve computational efficiency and robustness. Key changes include:

- **Non-Centered Parameterization**:
  - **Original**: Uses centered parameterization for random effects (`nu ~ multi_normal(zero, Sigma_nu)`).
  - **Modified**: Adopts non-centered parameterization with standardized random effects (`z[j] ~ normal(0, 1)`) and Cholesky factorization (`nu[j] = L_Sigma * z[j]`), improving MCMC sampling efficiency.
  
- **Covariance Structure**:
  - **Original**: Defines a covariance matrix `Sigma_nu` using `corr_matrix[2] omega_nu` and `quad_form_diag`.
  - **Modified**: Uses a Cholesky factor `cholesky_factor_corr[2] L_omega` and `L_Sigma = diag_pre_multiply(sigma_nu, L_omega)`, aligning with non-centered parameterization.
  
- **Priors**:
  - **Fixed Effects (`mu`)**: Changed from `normal(0, 1000)` to `normal(0, 2)` for more informative priors.
  - **Random Effects Variance (`sigma_nu`)**: Changed from `gamma(1.5, 1.0E-4)` to `normal(0, 0.5) T[0,]` for stability.
  - **Correlation**: Changed from `lkj_corr(2.0)` to `lkj_corr_cholesky(2.0)` for Cholesky factorization.
  - **Zero-Inflation (`pi`)**: Changed from `beta(0.5, 1.5)` to `beta(1, 1)` for a uniform prior.
  
- **Data Syntax**:
  - **Original**: Uses `int y[J,2]` and `int sample[J,2]`.
  - **Modified**: Uses `array[J,2] int y` and `array[J,2] int sample` for modern Stan syntax.
  
- **Generated Quantities**:
  - **Original**: Lacks extensive outputs.
  - **Modified**: Includes marginal probabilities (`p_control`, `p_treatment`), effect measures (Risk Ratio, Odds Ratio, Risk Difference), and posterior predictive samples (`y_rep`) for comprehensive analysis.

These changes enhance computational efficiency, prior robustness, and output interpretability while maintaining the model’s core structure.

## Results

The analysis produces:
- Posterior estimates of opioid effects on respiratory outcomes.
- Subgroup effects via meta-regression.
- Diagnostic plots (e.g., posterior predictive checks, trace plots).
- A summarizing figure (`04_results.qmd`) comparing overall and subgroup results.

Outputs are stored in `analyses/`. Fitted models in `models/storage/` can be reused with `cmdstanr`.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Open a pull request.

Ensure compatibility with `cmdstanr` and Quarto, and include documentation.

## Contact

For questions, contact Arthur Albuquerque via [GitHub](https://github.com/arthur-albuquerque) or open an issue in this repository.