import numpy as np

class CreditDefaultModel:
    def __init__(self, default_probabilities, correlations=None):
        """
        default_probabilities: array-like of marginal default probabilities for obligors.
        correlations: correlation matrix of obligors default dependence (e.g., Gaussian copula).
        """
        self.default_probabilities = np.array(default_probabilities)
        self.n = len(default_probabilities)
        if correlations is None:
            self.correlations = np.eye(self.n)
        else:
            self.correlations = correlations

    def simulate_defaults(self, size):
        """
        Simulate correlated default events using a Gaussian copula approach.
        Returns boolean default matrix: size x n obligors.
        """
        mean = np.zeros(self.n)
        L = np.linalg.cholesky(self.correlations)
        z = np.random.normal(size=(size, self.n))
        correlated_normals = z @ L.T
        uniforms = norm.cdf(correlated_normals)
        defaults = uniforms < self.default_probabilities
        return defaults

class ExposureProfile:
    def __init__(self, initial_exposures, exposure_paths=None):
        """
        initial_exposures: array-like, exposures at time 0.
        exposure_paths: function or None, maps number of steps -> exposure matrix (time x obligors).
        If None, exposures are static.
        """
        self.initial_exposures = np.array(initial_exposures)
        self.exposure_paths = exposure_paths

    def get_exposure(self, time_step=None, num_steps=None):
        """
        Returns exposure values for obligors at given time_step or generates a path over num_steps.
        """
        if self.exposure_paths is None:
            # Static exposure
            if time_step is not None:
                return self.initial_exposures
            elif num_steps is not None:
                return np.tile(self.initial_exposures, (num_steps, 1))
            else:
                return self.initial_exposures
        else:
            if num_steps is not None:
                return self.exposure_paths(num_steps)
            if time_step is not None:
                exposures = self.exposure_paths(1)
                return exposures[time_step]
            raise ValueError("Specify time_step or num_steps when exposure_paths is provided")

class CollateralModel:
    def __init__(self, collateral_amounts):
        """
        collateral_amounts: array-like collateral per obligor.
        """
        self.collateral_amounts = np.array(collateral_amounts)

    def apply_collateral(self, exposures):
        """
        Apply collateral to reduce exposures.
        Exposure is reduced by collateral but cannot go below zero.
        """
        net_exposure = exposures - self.collateral_amounts
        net_exposure[net_exposure < 0] = 0
        return net_exposure

class MarginCallModel:
    def __init__(self, margin_call_thresholds):
        """
        margin_call_thresholds: array-like thresholds per obligor.
        """
        self.thresholds = np.array(margin_call_thresholds)

    def check_margin_calls(self, exposures):
        """
        Given exposures, determine which obligors trigger margin calls.
        """
        return exposures > self.thresholds

class NettingSet:
    def __init__(self, obligor_groups):
        """
        obligor_groups: dict mapping netting set id -> list of obligor indices.
        """
        self.obligor_groups = obligor_groups

    def net_exposures(self, exposures):
        """
        exposures: array-like of exposures per obligor.
        Returns dict mapping netting set id -> netted exposure (sum over group).
        """
        netted = {}
        for netting_id, obligors in self.obligor_groups.items():
            netted[netting_id] = np.sum(exposures[obligors])
        return netted

class CVACalculator:
    def __init__(self, credit_default_model, exposure_profile, collateral_model=None,
                 margin_call_model=None, netting_set=None, discount_curve=None):
        self.cdm = credit_default_model
        self.exposure_profile = exposure_profile
        self.collateral_model = collateral_model
        self.margin_call_model = margin_call_model
        self.netting_set = netting_set
        self.discount_curve = discount_curve if discount_curve is not None else lambda t: 1.0

    def calculate_cva(self, num_simulations, time_horizon, time_steps):
        dt = time_horizon / time_steps

        default_events = self.cdm.simulate_defaults(num_simulations)  # shape (num_simulations, n obligors)

        exposures = self.exposure_profile.get_exposure(num_steps=time_steps)  # shape (time_steps, n obligors)

        cva_sum = 0.0

        for t in range(time_steps):
            discount_factor = self.discount_curve(t * dt)
            exposure_t = exposures[t]

            if self.collateral_model:
                exposure_t = self.collateral_model.apply_collateral(exposure_t)

            if self.margin_call_model:
                margin_calls = self.margin_call_model.check_margin_calls(exposure_t)
                # Simplified assumption: exposure reduced by 50% if margin call applies
                exposure_t[margin_calls] *= 0.5

            if self.netting_set:
                netted_exposures = self.netting_set.net_exposures(exposure_t)
                # Using max(net exposure, 0) for positive exposure only
                positive_net_exposure = {k: max(v, 0) for k, v in netted_exposures.items()}
                exposure_t = np.array([positive_net_exposure.get(k, 0) for k in sorted(self.netting_set.obligor_groups.keys())])
                # Adjust default_events accordingly - here assumed simplified approach

            default_mask = default_events[:, :]  # shape (num_simulations, obligors)

            exposure_at_defaults = default_mask * exposure_t  # broadcasting: exposures_t broadcast to (num_simulations,n)

            expected_loss_t = discount_factor * np.mean(np.sum(exposure_at_defaults, axis=1))

            cva_sum += expected_loss_t * dt

        return cva_sum
