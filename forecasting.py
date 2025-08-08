import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class HybridForecaster:
    def __init__(self, model_type='linear'):
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor()
        else:
            raise ValueError("Unsupported model type")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
