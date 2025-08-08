import pymc3 as pm
import arviz as az
import numpy as np

class BayesianCalibration:
    def __init__(self, model_func, observed_data, draws=2000, tune=1000, chains=4, target_accept=0.95):
        self.model_func = model_func
        self.observed_data = observed_data
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.model = None
        self.trace = None

    def calibrate(self):
        with pm.Model() as model:
            params = self.model_func()
            # Assuming model_func returns a dict of PyMC3 random variables representing parameters.
            # A simple normal likelihood for observations conditional on parameters.
            likelihood = pm.Normal('obs', mu=params['mu'], sigma=params['sigma'], observed=self.observed_data)
            self.trace = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains,
                                   target_accept=self.target_accept, return_inferencedata=True)
            self.model = model

    def summary(self):
        if self.trace is None:
            raise ValueError("Model is not calibrated yet.")
        return az.summary(self.trace)

    def traceplot(self):
        if self.trace is None:
            raise ValueError("Model is not calibrated yet.")
        return az.plot_trace(self.trace)

    def diagnostics(self):
        if self.trace is None:
            raise ValueError("Model is not calibrated yet.")
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        return {'rhat': rhat, 'ess': ess}

class HierarchicalBayesianCalibration:
    def __init__(self, model_func, observed_data, group_indices, draws=2000, tune=1000, chains=4, target_accept=0.95):
        """
        group_indices: array-like mapping each observed data point to a group index for hierarchical structure.
        """
        self.model_func = model_func
        self.observed_data = observed_data
        self.group_indices = np.array(group_indices)
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.model = None
        self.trace = None

    def calibrate(self):
        unique_groups = np.unique(self.group_indices)
        with pm.Model() as model:
            # Hyperpriors
            mu_mu = pm.Normal('mu_mu', mu=0, sigma=10)
            sigma_mu = pm.HalfNormal('sigma_mu', sigma=5)
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=5)

            # Group-level means
            mu_group = pm.Normal('mu_group', mu=mu_mu, sigma=sigma_mu, shape=len(unique_groups))

            # Map each observation to group mean
            mu_obs = mu_group[self.group_indices]

            # Likelihood
            obs = pm.Normal('obs', mu=mu_obs, sigma=sigma_obs, observed=self.observed_data)

            self.trace = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains,
                                   target_accept=self.target_accept, return_inferencedata=True)
            self.model = model

    def summary(self):
        if self.trace is None:
            raise ValueError("Model is not calibrated yet.")
        return az.summary(self.trace)

    def traceplot(self):
        if self.trace is None:
            raise ValueError("Model is not calibrated yet.")
        return az.plot_trace(self.trace)

    def diagnostics(self):
        if self.trace is None:
            raise ValueError("Model is not calibrated yet.")
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        return {'rhat': rhat, 'ess': ess}
