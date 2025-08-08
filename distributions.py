import numpy as np
from scipy.stats import norm, t, skellam, rv_continuous
from scipy.special import erf

class Normal:
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return norm.pdf(x, self.loc, self.scale)

    def cdf(self, x):
        return norm.cdf(x, self.loc, self.scale)

    def ppf(self, q):
        return norm.ppf(q, self.loc, self.scale)

    def sample(self, size):
        return np.random.normal(self.loc, self.scale, size)

class StudentT:
    def __init__(self, df, loc=0.0, scale=1.0):
        self.df = df
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return t.pdf(x, self.df, self.loc, self.scale)

    def cdf(self, x):
        return t.cdf(x, self.df, self.loc, self.scale)

    def ppf(self, q):
        return t.ppf(q, self.df, self.loc, self.scale)

    def sample(self, size):
        return np.random.standard_t(self.df, size) * self.scale + self.loc

class SkewedNormal(rv_continuous):
    def __init__(self, loc=0, scale=1, alpha=0):
        super().__init__(name='skewnorm')
        self.loc = loc
        self.scale = scale
        self.alpha = alpha

    def _pdf(self, x):
        t = (x - self.loc) / self.scale
        return 2/self.scale * norm.pdf(t) * norm.cdf(self.alpha * t)

    def sample(self, size):
        delta = self.alpha / np.sqrt(1 + self.alpha**2)
        u0 = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        u1 = delta * u0 + np.sqrt(1 - delta**2) * v
        samples = u0.copy()
        samples[u1 < 0] = -samples[u1 < 0]
        return samples * self.scale + self.loc

class Mixture:
    def __init__(self, components, weights):
        self.components = components
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()

    def pdf(self, x):
        return sum(w * comp.pdf(x) for comp, w in zip(self.components, self.weights))

    def cdf(self, x):
        return sum(w * comp.cdf(x) for comp, w in zip(self.components, self.weights))

    def sample(self, size):
        samples = np.zeros(size)
        n_components = len(self.components)
        component_indices = np.random.choice(n_components, size=size, p=self.weights)
        for i, comp in enumerate(self.components):
            idx = np.where(component_indices == i)[0]
            if idx.size > 0:
                samples[idx] = comp.sample(len(idx))
        return samples
