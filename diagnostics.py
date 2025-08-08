import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.linear_model import OLS

class ResidualAnalysis:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.residuals = None

    def calculate_residuals(self):
        self.residuals = self.data - self.model.predict(self.data)
        return self.residuals

    def heteroscedasticity_test(self, exog):
        if self.residuals is None:
            self.calculate_residuals()
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(self.residuals, exog)
        return {'lm_stat': lm_stat, 'lm_pvalue': lm_pvalue, 'f_stat': f_stat, 'f_pvalue': f_pvalue}

class InfluenceMeasures:
    def __init__(self, model):
        self.model = model

    def get_influence(self):
        influence = self.model.get_influence()
        summary = influence.summary_frame()
        return summary

class HeteroscedasticityTests:
    @staticmethod
    def breit_pagan_test(residuals, exog):
        return het_breuschpagan(residuals, exog)
