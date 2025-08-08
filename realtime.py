import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

class IncrementalUpdater:
    def __init__(self, base_model=None):
        """
        base_model: sklearn-like model supporting partial_fit or fit. 
                    If None, RandomForestRegressor is used but note RF does not support partial_fit,
                    so a workaround with retraining on growing dataset is used.
        """
        self.base_model = base_model if base_model is not None else RandomForestRegressor()
        self.X_data = None
        self.y_data = None
        self.is_fitted = False
        self.model = clone(self.base_model)

    def update(self, X_new, y_new):
        """
        Update the model incrementally with new data.
        For models with partial_fit, use it. Otherwise, retrain on cumulative data,
        which may be costly but necessary for models without partial_fit.
        """
        if self.X_data is None:
            self.X_data = X_new
            self.y_data = y_new
        else:
            self.X_data = np.vstack([self.X_data, X_new])
            self.y_data = np.concatenate([self.y_data, y_new])

        if hasattr(self.model, 'partial_fit'):
            if not self.is_fitted:
                self.model.partial_fit(X_new, y_new)
                self.is_fitted = True
            else:
                self.model.partial_fit(X_new, y_new)
        else:
            # Retrain on full data (cumulative)
            self.model.fit(self.X_data, self.y_data)
            self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call update() first.")
        return self.model.predict(X)

class AISimulationOptimizer:
    def __init__(self, budget, model=None):
        """
        budget: total number of simulations or resources available for allocation.
        model: Optional AI model to optimize simulation allocation.
               Should have fit() and predict() interfaces.
        """
        self.budget = budget
        self.model = model
        self.current_allocation = None

    def allocate(self, simulation_results_features):
        """
        Allocate simulations intelligently based on features from simulation results.
        If model is not provided, equal allocation is used.
        simulation_results_features: np.array of shape (n_scenarios, n_features) representing scenario features
        """
        if self.model is None:
            n = simulation_results_features.shape[0]
            allocation = np.full(n, self.budget / n)
            self.current_allocation = allocation
            return allocation
        else:
            allocation_scores = self.model.predict(simulation_results_features)
            allocation_scores = np.maximum(allocation_scores, 0)  # no negative allocations
            total_score = np.sum(allocation_scores)
            if total_score == 0:
                # fallback in case model predicts zero allocations
                n = simulation_results_features.shape[0]
                allocation = np.full(n, self.budget / n)
            else:
                allocation = allocation_scores / total_score * self.budget
            self.current_allocation = allocation
            return allocation

    def train(self, X, y):
        """
        Train the underlying AI model.
        X: feature matrix from previous simulation data.
        y: target allocation or performance metric to optimize.
        """
        if self.model is None:
            raise ValueError("No AI model specified for optimization.")
        self.model.fit(X, y)

    def update_model(self, X_new, y_new):
        """
        Incrementally update the AI model if it supports partial_fit,
        otherwise retrain on cumulative data.
        """
        if self.model is None:
            raise ValueError("No AI model specified for optimization.")

        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
        else:
            # If no incremental fit, retrain on accumulated data (not implemented here)
            # Raise error or implement retraining strategy outside
            raise NotImplementedError("Model does not support partial_fit and incremental update not implemented.")

