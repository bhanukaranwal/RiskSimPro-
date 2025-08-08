import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierDetector:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def detect_outliers(self, X):
        preds = self.predict(X)
        # Outliers are labeled -1 by IsolationForest
        outliers = np.where(preds == -1)[0]
        return outliers
