import numpy as np

class CreditDefaultModel:
    def __init__(self, default_probabilities):
        self.default_probabilities = default_probabilities

    def simulate_defaults(self, size):
        defaults = np.random.uniform(size=size) < self.default_probabilities
        return defaults

class ExposureModel:
    def __init__(self, exposures):
        self.exposures = exposures

    def get_expected_exposures(self):
        return np.array(self.exposures)

class CVA:
    def __init__(self, default_model, exposure_model, discount_factor=1.0):
        self.default_model = default_model
        self.exposure_model = exposure_model
        self.discount_factor = discount_factor

    def calculate(self, num_simulations):
        defaults = self.default_model.simulate_defaults(num_simulations)
        exposures = np.tile(self.exposure_model.get_expected_exposures(), (num_simulations, 1))
        losses = np.where(defaults[:, None], exposures, 0)
        expected_loss = self.discount_factor * losses.mean(axis=0)
        return expected_loss

class Margining:
    def __init__(self, margin_call_threshold):
        self.margin_call_threshold = margin_call_threshold

    def check_margin_call(self, exposure):
        return exposure > self.margin_call_threshold

class Netting:
    def __init__(self, netting_groups):
        self.netting_groups = netting_groups

    def net_exposures(self, exposures):
        netted = {}
        for group, indices in self.netting_groups.items():
            netted[group] = sum(exposures[i] for i in indices)
        return netted
