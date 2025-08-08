import numpy as np
from arch import arch_model

class GARCH:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.fit_result = None

    def fit(self, returns):
        self.model = arch_model(returns, vol='Garch', p=self.p, q=self.q, dist='normal')
        self.fit_result = self.model.fit(disp='off')

    def simulate(self, horizon):
        return self.fit_result.simulate(self.fit_result.params, horizon).data['volatility']

class EGARCH:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.fit_result = None

    def fit(self, returns):
        self.model = arch_model(returns, vol='EGarch', p=self.p, q=self.q, dist='normal')
        self.fit_result = self.model.fit(disp='off')

    def simulate(self, horizon):
        return self.fit_result.simulate(self.fit_result.params, horizon).data['volatility']

class GJRGARCH:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.fit_result = None

    def fit(self, returns):
        self.model = arch_model(returns, vol='GJRGarch', p=self.p, q=self.q, dist='normal')
        self.fit_result = self.model.fit(disp='off')

    def simulate(self, horizon):
        return self.fit_result.simulate(self.fit_result.params, horizon).data['volatility']

class StochasticVolatility:
    def __init__(self, mu=0, phi=0.95, sigma=0.1):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.volatility = None

    def simulate(self, n):
        vol = np.zeros(n)
        vol[0] = self.mu
        for t in range(1, n):
            vol[t] = self.mu + self.phi * (vol[t-1] - self.mu) + self.sigma * np.random.normal()
        self.volatility = np.exp(vol/2)
        return self.volatility

class RoughVolatility:
    def __init__(self, H=0.1, sigma=0.1):
        self.H = H
        self.sigma = sigma

    def simulate(self, n):
        # Placeholder for rough volatility simulation (e.g., using fractional Brownian motion)
        return np.zeros(n)

class RegimeSwitchingVolatility:
    def __init__(self, regimes, transition_matrix, initial_state=0):
        self.regimes = regimes
        self.transition_matrix = transition_matrix
        self.current_state = initial_state

    def simulate(self, n):
        vol = np.zeros(n)
        for t in range(n):
            vol[t] = self.regimes[self.current_state].simulate(1)[0]
            self.current_state = np.random.choice(len(self.regimes), p=self.transition_matrix[self.current_state])
        return vol
