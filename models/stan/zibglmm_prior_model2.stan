data {
  int<lower=0> J;                  // number of studies 
  vector[2] zero;
  array[J,2] int sample;           // sample sizes (used for simulating data)
}

parameters {
  vector[2] mu;                    // fixed effects for control and treatment
  vector<lower=0>[2] sigma_nu;     // standard deviations for random effects
  cholesky_factor_corr[2] L_omega; // Cholesky factor of correlation matrix
  array[J] vector[2] z;            // Standardized random effects (non-centered)
  real<lower=0, upper=1> pi;       // zero inflation rate
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
  // Priors only (no likelihood)
  mu ~ normal(0, 10);
  sigma_nu ~ normal(0, 0.5) T[0,];
  L_omega ~ lkj_corr_cholesky(2.0); 
  pi ~ beta(1, 1);

  // Standardized random effects
  for (j in 1:J) {
    z[j] ~ normal(0, 1);          // Standard normal
  }
}

generated quantities {
  real C = 16 * sqrt(3) / (15 * pi()); // Constant C ≈ 0.5887
  real p_control;                      // Marginal probability for control (E(p_0))
  real p_treatment;                    // Marginal probability for treatment (E(p_1))
  real RR;                             // Marginal risk ratio for at-risk population
  real OR;                             // Marginal odds ratio for at-risk population
  real RD;                             // Marginal risk difference for at-risk population
  array[J,2] int y_prior;              // Prior predictive samples


  // Compute marginal probabilities for the at-risk population
  p_control = inv_logit(mu[1] / sqrt(1 + C^2 * sigma_nu[1]^2));
  p_treatment = inv_logit(mu[2] / sqrt(1 + C^2 * sigma_nu[2]^2));

  // Compute marginal risk ratio for the at-risk population
  RR = p_treatment / p_control;
  
  // Compute marginal odds ratio for the at-risk population
  OR = (p_treatment / (1 - p_treatment)) / (p_control / (1 - p_control));
  
  // Compute marginal risk difference for the at-risk population
  RD = p_treatment - p_control;
  
  // Prior predictive sampling
  for (n in 1:J) {
    int is_zero_inflated = bernoulli_rng(pi);
    if (is_zero_inflated) {
      y_prior[n,1] = 0;
      y_prior[n,2] = 0;
    } else {
      real p_control_n = inv_logit(mu[1] + nu[n,1]);
      real p_treatment_n = inv_logit(mu[2] + nu[n,2]);
      y_prior[n,1] = binomial_rng(sample[n,1], p_control_n);
      y_prior[n,2] = binomial_rng(sample[n,2], p_treatment_n);
    }
  }
}

