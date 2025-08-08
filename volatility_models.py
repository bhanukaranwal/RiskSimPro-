import numpy as np
from arch import arch_model
import pymc3 as pm

class GARCH:
    def __init__(self, p=1, q=1, dist='normal'):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.fit_result = None

    def fit(self, returns):
        self.model = arch_model(returns, vol="Garch", p=self.p, q=self.q, dist=self.dist)
        self.fit_result = self.model.fit(disp="off")
        return self.fit_result

    def simulate(self, horizon):
        if self.fit_result is None:
            raise ValueError("Model must be fit before simulation")
        sim = self.fit_result.simulate(self.fit_result.params, horizon)
        return sim.data["volatility"]

class EGARCH:
    def __init__(self, p=1, q=1, dist='normal'):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.fit_result = None

    def fit(self, returns):
        self.model = arch_model(returns, vol="EGarch", p=self.p, q=self.q, dist=self.dist)
        self.fit_result = self.model.fit(disp="off")
        return self.fit_result

    def simulate(self, horizon):
        if self.fit_result is None:
            raise ValueError("Model must be fit before simulation")
        sim = self.fit_result.simulate(self.fit_result.params, horizon)
        return sim.data["volatility"]

class GJRGARCH:
    def __init__(self, p=1, q=1, dist='normal'):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.fit_result = None

    def fit(self, returns):
        self.model = arch_model(returns, vol="GARCH", p=self.p, q=self.q, power=2.0, o=1, dist=self.dist)
        self.fit_result = self.model.fit(disp="off")
        return self.fit_result

    def simulate(self, horizon):
        if self.fit_result is None:
            raise ValueError("Model must be fit before simulation")
        sim = self.fit_result.simulate(self.fit_result.params, horizon)
        return sim.data["volatility"]

class StochasticVolatility:
    def __init__(self, mu=0, phi=0.98, sigma=0.15):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.model = None
        self.trace = None

    def simulate(self, T):
        h = np.zeros(T)
        y = np.zeros(T)
        h[0] = self.mu
        y[0] = np.exp(h[0]/2) * np.random.normal()
        for t in range(1, T):
            h[t] = self.mu + self.phi * (h[t-1] - self.mu) + self.sigma * np.random.normal()
            y[t] = np.exp(h[t]/2) * np.random.normal()
        return y, np.exp(h/2)

    def bayesian_fit(self, returns, draws=1000, chains=2):
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=10)
            phi = pm.Beta('phi', alpha=20, beta=1.5)
            sigma = pm.Exponential('sigma', 1.0)
            h = pm.GaussianRandomWalk('h', mu=mu, sigma=sigma, shape=len(returns))
            obs = pm.Normal('y', mu=0, sigma=pm.math.exp(h/2), observed=returns)
            trace = pm.sample(draws=draws, chains=chains, progressbar=False)
        self.model = model
        self.trace = trace
        return trace

class RoughVolatility:
    def __init__(self, H=0.06, eta=0.3, xi=0.21, T=1.0):
        self.H = H   # Hurst exponent
        self.eta = eta
        self.xi = xi
        self.T = T

    def fbm(self, n):
        dt = self.T / n
        times = np.linspace(0, self.T, n)
        cov = lambda s, t: 0.5 * (s**(2*self.H) + t**(2*self.H) - abs(s-t)**(2*self.H))
        K = np.fromfunction(
            np.vectorize(lambda i, j: cov(times[int(i)], times[int(j)])), (n, n), dtype=int)
        L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
        z = np.random.normal(size=n)
        return np.dot(L, z)

    def simulate(self, n):
        # Log-volatility: X_t = xi * FBM(H, t) - 0.5 * eta**2
        fbm_path = self.fbm(n)
        log_vol = self.xi * fbm_path - 0.5 * self.eta**2
        vol = np.exp(log_vol)
        return vol
