data {
  int<lower=0> J;                  // number of studies 
  vector[2] zero;
  array[J,2] int sample;           // sample sizes (used for simulating data)
}

parameters {
  vector[2] mu;                    // fixed effects for treatment and control
  vector<lower=0>[2] sigma_nu;     // standard deviations for random effects
  corr_matrix[2] omega_nu;         // correlation matrix for random effects
  array[J] vector[2] nu;           // control random effects (centered parameterization)
  real<lower=0, upper=1> pi;       // zero inflation rate
}

transformed parameters {
  cov_matrix[2] Sigma_nu;
  Sigma_nu = quad_form_diag(omega_nu, sigma_nu);  // Covariance matrix for random effects
}

model {
  // Priors only (no likelihood)
  mu ~ normal(0, 1000);
  sigma_nu ~ gamma(1.5, 1.0E-4);
  omega_nu ~ lkj_corr(2.0);
  pi ~ beta(0.5, 1.5);
  
  // Random effects prior
  for (j in 1:J) {
    nu[j] ~ multi_normal(zero, Sigma_nu);
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
    int is_zero_inflated = bernoulli_rng(pi);  // Sample zero-inflation
    if (is_zero_inflated) {
      y_prior[n,1] = 0;
      y_prior[n,2] = 0;
    } else {
      real p_control_n = inv_logit(mu[1] + nu[n,1]);  // Probability for control
      real p_treatment_n = inv_logit(mu[2] + nu[n,2]);  // Probability for treatment
      y_prior[n,1] = binomial_rng(sample[n,1], p_control_n);  // Simulate control events
      y_prior[n,2] = binomial_rng(sample[n,2], p_treatment_n);  // Simulate treatment events
    }
  }
}

