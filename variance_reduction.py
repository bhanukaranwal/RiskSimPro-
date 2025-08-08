import numpy as np
from scipy.stats import norm

class AdaptiveImportanceSampling:
    def __init__(self, target_pdf, initial_proposal_mu=0.0, initial_proposal_sigma=1.0, max_iter=10, tol=1e-3):
        self.target_pdf = target_pdf  # function: f(x)
        self.proposal_mu = initial_proposal_mu
        self.proposal_sigma = initial_proposal_sigma
        self.max_iter = max_iter
        self.tol = tol

    def proposal_pdf(self, x):
        return norm.pdf(x, self.proposal_mu, self.proposal_sigma)

    def sample_proposal(self, size):
        return np.random.normal(self.proposal_mu, self.proposal_sigma, size)

    def update_proposal(self, samples, weights):
        weighted_mean = np.sum(samples * weights) / np.sum(weights)
        weighted_var = np.sum(weights * (samples - weighted_mean)**2) / np.sum(weights)
        # Update proposal parameters with damping to prevent instability
        alpha = 0.7
        new_mu = alpha * weighted_mean + (1 - alpha) * self.proposal_mu
        new_sigma = alpha * np.sqrt(weighted_var) + (1 - alpha) * self.proposal_sigma
        return new_mu, new_sigma

    def run(self, num_samples=10000):
        self.proposal_mu = 0.0
        self.proposal_sigma = 1.0
        prev_mu, prev_sigma = None, None

        for iteration in range(self.max_iter):
            samples = self.sample_proposal(num_samples)
            weights = self.target_pdf(samples) / self.proposal_pdf(samples)
            weights /= np.sum(weights)  # normalize weights
            new_mu, new_sigma = self.update_proposal(samples, weights)

            # Check convergence
            if prev_mu is not None and prev_sigma is not None:
                mu_change = abs(new_mu - prev_mu)
                sigma_change = abs(new_sigma - prev_sigma)
                if mu_change < self.tol and sigma_change < self.tol:
                    break
            self.proposal_mu, self.proposal_sigma = new_mu, new_sigma
            prev_mu, prev_sigma = new_mu, new_sigma

        # Final weighted samples and weights for estimation
        samples = self.sample_proposal(num_samples)
        weights = self.target_pdf(samples) / self.proposal_pdf(samples)
        weights /= np.sum(weights)
        return samples, weights

class StratifiedSamplingOptimal:
    def __init__(self, distribution_ppf, strata=20):
        self.ppf = distribution_ppf
        self.strata = strata

    def sample(self, size):
        samples = np.zeros(size)
        strata_size = size // self.strata
        remainder = size % self.strata

        # Allocate samples proportionally based on strata variance (placeholder: equal here)
        for i in range(self.strata):
            u = np.random.uniform(i/self.strata, (i+1)/self.strata, strata_size)
            samples[i*strata_size:(i+1)*strata_size] = self.ppf(u)

        if remainder > 0:
            u = np.random.uniform(0, 1, remainder)
            samples[-remainder:] = self.ppf(u)

        return samples

class ControlVariatesOptimized:
    def __init__(self, control_variate_func, expected_value):
        self.control_variate_func = control_variate_func
        self.expected_value = expected_value
        self.beta = None

    def fit(self, estimates, control_variate_samples):
        cov = np.cov(estimates, control_variate_samples)
        if cov.shape == (2, 2) and cov[1,1] != 0:
            self.beta = cov[0, 1] / cov[1, 1]
        else:
            self.beta = 0.0

    def adjust(self, estimates, control_variate_samples):
        if self.beta is None:
            self.fit(estimates, control_variate_samples)
        adjusted = estimates - self.beta * (control_variate_samples - self.expected_value)
        return adjusted

class AntitheticVariatesOptimized:
    def __init__(self, distribution_sampler):
        self.distribution_sampler = distribution_sampler

    def sample(self, size):
        half_size = size // 2
        samples1 = self.distribution_sampler(half_size)
        samples2 = -samples1
        samples = np.concatenate([samples1, samples2])
        if size % 2 == 1:
            extra_sample = self.distribution_sampler(1)
            samples = np.append(samples, extra_sample)
        return samples

class LatinHypercubeSampling:
    def __init__(self, dimension):
        self.dimension = dimension

    def sample(self, size):
        segments = np.linspace(0, 1, size + 1)
        u = np.random.uniform(low=segments[:-1], high=segments[1:], size=(size, self.dimension))
        for j in range(self.dimension):
            np.random.shuffle(u[:, j])
        return u
