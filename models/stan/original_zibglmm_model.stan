data {
  int<lower=0> J;                  // number of studies 
  vector[2] zero;
  int y[J,2];
  int sample[J,2];
}
parameters {
  vector[2] mu;         // fixed effects for treatment and control
  vector<lower=0>[2] sigma_nu; // control variance
  corr_matrix[2] omega_nu; // correlation matrix for random effects
  vector[2] nu[J];         // control random effects
  real<lower=0, upper=1> pi; // zero inflation rate
}
transformed parameters {
  cov_matrix[2] Sigma_nu;
  Sigma_nu = quad_form_diag(omega_nu, sigma_nu);
}
model{
  nu ~ multi_normal(zero, Sigma_nu);
  sigma_nu ~ gamma(1.5, 1.0E-4);
  omega_nu ~ lkj_corr(2.0);
  mu ~ normal(0, 1000);
  pi ~ beta(0.5,1.5);
  for(n in 1:J){
    if(y[n,1] == 0 && y[n,2]==0){
      target += log_sum_exp(bernoulli_lpmf(1|pi),
                      bernoulli_logit_lpmf(0|pi)  + 
      binomial_logit_lpmf(y[n,1] | sample[n,1], mu[1] + nu[n,1]) + 
      binomial_logit_lpmf(y[n,2] | sample[n,2], mu[2] + nu[n,2] ));
    }
    else{
      target += (bernoulli_lpmf(0|pi) + 
      binomial_logit_lpmf(y[n,1] | sample[n,1], mu[1] + nu[n,1]) + 
      binomial_logit_lpmf(y[n,2] | sample[n,2], mu[2] + nu[n,2]));
    }
  }
}