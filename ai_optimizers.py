import numpy as np

class AISimulationOptimizer:
    def __init__(self, budget, model=None):
        self.budget = budget
        self.model = model
        self.current_allocation = None

    def allocate(self, simulation_results):
        if self.model is None:
            # Simple equal allocation as fallback
            allocation = np.full(len(simulation_results), self.budget / len(simulation_results))
            self.current_allocation = allocation
            return allocation
        else:
            # Placeholder for AI model-based allocation logic
            features = self.extract_features(simulation_results)
            allocation = self.model.predict(features)
            allocation = self.normalize_allocation(allocation)
            self.current_allocation = allocation
            return allocation

    def extract_features(self, simulation_results):
        # Placeholder for feature extraction from simulation results
        mean = np.mean(simulation_results, axis=1)
        std = np.std(simulation_results, axis=1)
        return np.column_stack((mean, std))

    def normalize_allocation(self, allocation):
        total = np.sum(allocation)
        if total > 0:
            return allocation / total * self.budget
        else:
            return np.full(len(allocation), self.budget / len(allocation))
