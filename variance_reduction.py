import numpy as np

class ImportanceSampling:
    def __init__(self, proposal_dist, target_dist):
        self.proposal_dist = proposal_dist
        self.target_dist = target_dist

    def sample(self, size):
        samples = self.proposal_dist.sample(size)
        weights = self.target_dist.pdf(samples) / self.proposal_dist.pdf(samples)
        return samples, weights

class StratifiedSampling:
    def __init__(self, distribution, strata=10):
        self.distribution = distribution
        self.strata = strata

    def sample(self, size):
        samples = np.zeros(size)
        strata_size = size // self.strata
        for i in range(self.strata):
            u = np.random.uniform(i/self.strata, (i+1)/self.strata, strata_size)
            samples[i*strata_size:(i+1)*strata_size] = self.distribution.ppf(u)
        remainder = size % self.strata
        if remainder > 0:
            u = np.random.uniform(0, 1, remainder)
            samples[-remainder:] = self.distribution.ppf(u)
        return samples

class QuasiMonteCarlo:
    def __init__(self, distribution, dim):
        self.distribution = distribution
        self.dim = dim

    def sample(self, size):
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        u = sampler.random(size)
        return self.distribution.ppf(u)

class ControlVariates:
    def __init__(self, control_variate_func, expected_value):
        self.control_variate_func = control_variate_func
        self.expected_value = expected_value

    def adjust(self, estimates, control_variate_samples):
        cov_matrix = np.cov(estimates, control_variate_samples)
        beta = cov_matrix[0,1] / cov_matrix[1,1]
        adjusted_estimate = estimates - beta * (control_variate_samples - self.expected_value)
        return adjusted_estimate

class AntitheticVariates:
    def __init__(self, distribution):
        self.distribution = distribution

    def sample(self, size):
        half_size = size // 2
        samples1 = self.distribution.sample(half_size)
        samples2 = -samples1
        return np.concatenate([samples1, samples2])
