import numpy as np
from scipy.stats import norm, t

class Normal:
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def sample(self, size):
        return np.random.normal(self.loc, self.scale, size)

class StudentT:
    def __init__(self, df, loc=0.0, scale=1.0):
        self.df = df
        self.loc = loc
        self.scale = scale

    def sample(self, size):
        return self.loc + self.scale * np.random.standard_t(self.df, size)

class Mixture:
    def __init__(self, components, weights):
        self.components = components
        self.weights = weights / np.sum(weights)

    def sample(self, size):
        n_components = len(self.components)
        samples = np.zeros(size)
        component_choices = np.random.choice(n_components, size=size, p=self.weights)
        for i, component in enumerate(self.components):
            count = np.sum(component_choices == i)
            if count > 0:
                samples[component_choices == i] = component.sample(count)
        return samples
