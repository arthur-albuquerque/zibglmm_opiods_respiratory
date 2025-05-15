data {
  int<lower=0> J;                  // number of studies 
  vector[2] zero;
  array[J,2] int sample;           // sample sizes
  array[J,2] int y;                // number of events
  array[J] int<lower=0, upper=1> subgroup; // Binary covariate for subgroup (0 or 1)
}

parameters {
  vector[2] mu;               // Baseline fixed effects for control and treatment
  vector[2] beta;                  // Coefficients for subgroup effect on control and treatment
  vector<lower=0>[2] sigma_nu;     // Standard deviations for random effects
  cholesky_factor_corr[2] L_omega; // Cholesky factor of correlation matrix
  array[J] vector[2] z;            // Standardized random effects (non-centered)
  real<lower=0, upper=1> pi;       // Zero inflation rate
}

transformed parameters {
  matrix[2, 2] L_Sigma;            // Cholesky factor of covariance matrix
  array[J] vector[2] nu;           // Actual random effects

  // Construct the Cholesky factor of the covariance matrix
  L_Sigma = diag_pre_multiply(sigma_nu, L_omega);

  // Compute actual random effects using non-centered parameterization
  for (j in 1:J) {
    nu[j] = L_Sigma * z[j];
  }
}

model {
  // Priors
  mu ~ normal(0, 2);          // Prior for baseline fixed effects
  beta ~ normal(0, 3);             // Prior for subgroup effect (vague)
  sigma_nu ~ normal(0, 0.5) T[0,]; // Prior for random effect standard deviations
  L_omega ~ lkj_corr_cholesky(2.0); 
  pi ~ beta(1, 1);

  // Standardized random effects
  for (j in 1:J) {
    z[j] ~ normal(0, 1);          // Standard normal
  }

  // Likelihood
  for (n in 1:J) {
    if (y[n,1] == 0 && y[n,2] == 0) {
      target += log_sum_exp(
        bernoulli_lpmf(1 | pi),
        bernoulli_lpmf(0 | pi) + 
        binomial_logit_lpmf(y[n,1] | sample[n,1], mu[1] + beta[1] * subgroup[n] + nu[n,1]) + 
        binomial_logit_lpmf(y[n,2] | sample[n,2], mu[2] + beta[2] * subgroup[n] + nu[n,2])
      );
    } else {
      target += bernoulli_lpmf(0 | pi) + 
               binomial_logit_lpmf(y[n,1] | sample[n,1], mu[1] + beta[1] * subgroup[n] + nu[n,1]) + 
               binomial_logit_lpmf(y[n,2] | sample[n,2], mu[2] + beta[2] * subgroup[n] + nu[n,2]);
    }
  }
}

generated quantities {
  real C = 16 * sqrt(3) / (15 * pi()); // Constant C ≈ 0.5887
  
  // Marginal probabilities for each subgroup
  real p_control_sub0;                 // Marginal probability for control, subgroup 0
  real p_control_sub1;                 // Marginal probability for control, subgroup 1
  real p_treatment_sub0;               // Marginal probability for treatment, subgroup 0
  real p_treatment_sub1;               // Marginal probability for treatment, subgroup 1
  
  // Subgroup-specific risk ratios, odds ratios, and risk differences
  real RR_sub0;                        // Risk ratio for subgroup 0
  real RR_sub1;                        // Risk ratio for subgroup 1
  real OR_sub0;                        // Odds ratio for subgroup 0
  real OR_sub1;                        // Odds ratio for subgroup 1
  real RD_sub0;                        // Risk difference for subgroup 0
  real RD_sub1;                        // Risk difference for subgroup 1
  
  array[J,2] int y_rep;                // Posterior predictive samples

  // Compute marginal probabilities for each subgroup
  p_control_sub0 = inv_logit(mu[1] / sqrt(1 + C^2 * sigma_nu[1]^2));          // Subgroup 0
  p_control_sub1 = inv_logit((mu[1] + beta[1]) / sqrt(1 + C^2 * sigma_nu[1]^2)); // Subgroup 1
  p_treatment_sub0 = inv_logit(mu[2] / sqrt(1 + C^2 * sigma_nu[2]^2));        // Subgroup 0
  p_treatment_sub1 = inv_logit((mu[2] + beta[2]) / sqrt(1 + C^2 * sigma_nu[2]^2)); // Subgroup 1

  // Compute subgroup-specific risk ratios
  RR_sub0 = p_treatment_sub0 / p_control_sub0;
  RR_sub1 = p_treatment_sub1 / p_control_sub1;
  
  // Compute subgroup-specific odds ratios
  OR_sub0 = (p_treatment_sub0 / (1 - p_treatment_sub0)) / (p_control_sub0 / (1 - p_control_sub0));
  OR_sub1 = (p_treatment_sub1 / (1 - p_treatment_sub1)) / (p_control_sub1 / (1 - p_control_sub1));
  
  // Compute subgroup-specific risk differences
  RD_sub0 = p_treatment_sub0 - p_control_sub0;
  RD_sub1 = p_treatment_sub1 - p_control_sub1;
  
  // Posterior predictive sampling
  for (n in 1:J) {
    int is_zero_inflated = bernoulli_rng(pi);  // Sample zero-inflation
    if (is_zero_inflated) {
      y_rep[n,1] = 0;
      y_rep[n,2] = 0;
    } else {
      real p_control_n = inv_logit(mu[1] + beta[1] * subgroup[n] + nu[n,1]);  // Adjusted for subgroup
      real p_treatment_n = inv_logit(mu[2] + beta[2] * subgroup[n] + nu[n,2]);  // Adjusted for subgroup
      y_rep[n,1] = binomial_rng(sample[n,1], p_control_n);  // Simulate control events
      y_rep[n,2] = binomial_rng(sample[n,2], p_treatment_n);  // Simulate treatment events
    }
  }
}

