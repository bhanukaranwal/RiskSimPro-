import numpy as np
from scipy.optimize import minimize

class CVaRPortfolioOptimizer:
    def __init__(self, returns, alpha=0.95, transaction_costs=None, liquidity_constraints=None):
        self.returns = returns
        self.alpha = alpha
        self.transaction_costs = transaction_costs
        self.liquidity_constraints = liquidity_constraints
        self.num_assets = returns.shape[1]

    def portfolio_loss(self, weights):
        portfolio_returns = self.returns @ weights
        var = np.quantile(-portfolio_returns, 1 - self.alpha)
        cvar = -portfolio_returns[portfolio_returns <= -var].mean()
        return cvar

    def optimize(self, initial_weights=None):
        if initial_weights is None:
            initial_weights = np.ones(self.num_assets) / self.num_assets

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(self.num_assets)]

        result = minimize(self.portfolio_loss, initial_weights,
                          method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else None

class EfficientFrontierExplorer:
    def __init__(self, returns):
        self.returns = returns
        self.num_assets = returns.shape[1]

    def mean_variance(self, weights):
        mean_return = np.mean(self.returns @ weights)
        variance = np.var(self.returns @ weights)
        return mean_return, variance

    def explore(self, num_points=50):
        weights_list = []
        returns_list = []
        variances_list = []

        for target_return in np.linspace(np.min(np.mean(self.returns, axis=0)),
                                         np.max(np.mean(self.returns, axis=0)), num_points):
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                           {'type': 'eq', 'fun': lambda x: np.mean(self.returns @ x) - target_return}]
            bounds = [(0, 1) for _ in range(self.num_assets)]
            initial_weights = np.ones(self.num_assets) / self.num_assets

            result = minimize(lambda w: np.var(self.returns @ w), initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                weights_list.append(result.x)
                mean_ret, var = self.mean_variance(result.x)
                returns_list.append(mean_ret)
                variances_list.append(var)

        return weights_list, returns_list, variances_list
