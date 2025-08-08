import numpy as np

def VaR(losses, confidence_level=0.95):
    return np.quantile(losses, 1 - confidence_level)

def CVaR(losses, confidence_level=0.95):
    var = VaR(losses, confidence_level)
    tail_losses = losses[losses >= var]
    return tail_losses.mean() if len(tail_losses) > 0 else var

class Backtesting:
    def __init__(self, actual_losses, predicted_var, alpha=0.05):
        self.actual_losses = actual_losses
        self.predicted_var = predicted_var
        self.alpha = alpha

    def kupiec_test(self):
        violations = self.actual_losses > self.predicted_var
        n_violations = np.sum(violations)
        n = len(self.actual_losses)
        pi = self.alpha
        p_hat = n_violations / n
        from scipy.stats import chi2
        lr_stat = -2 * (np.log((1-pi)**(n-n_violations) * pi**n_violations) -
                        np.log((1-p_hat)**(n-n_violations) * p_hat**n_violations))
        p_value = 1 - chi2.cdf(lr_stat, df=1)
        return lr_stat, p_value

class ConvergenceDiagnostics:
    def __init__(self, simulation_results):
        self.simulation_results = simulation_results

    def relative_error(self):
        mean = np.mean(self.simulation_results)
        std = np.std(self.simulation_results)
        return std / mean if mean != 0 else np.inf
