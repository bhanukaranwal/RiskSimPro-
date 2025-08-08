import pymc3 as pm
import numpy as np

class BayesianCalibration:
    def __init__(self, model_func, observed_data):
        self.model_func = model_func
        self.observed_data = observed_data
        self.trace = None

    def calibrate(self):
        with pm.Model() as model:
            params = self.model_func()
            likelihood = pm.Normal('obs', mu=params, sigma=1, observed=self.observed_data)
            self.trace = pm.sample(return_inferencedata=False)

    def get_posterior_means(self):
        return {var: self.trace[var].mean() for var in self.trace.varnames if not var.endswith('__')}

class RollingCalibration:
    def __init__(self, model_func, data, window_size):
        self.model_func = model_func
        self.data = data
        self.window_size = window_size
        self.posteriors = []

    def calibrate(self):
        for start in range(len(self.data) - self.window_size + 1):
            window_data = self.data[start:start+self.window_size]
            calib = BayesianCalibration(self.model_func, window_data)
            calib.calibrate()
            self.posteriors.append(calib.get_posterior_means())
        return self.posteriors
