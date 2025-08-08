import unittest
import numpy as np

from risksimpro.distributions import Normal, StudentT, Mixture
from risksimpro.copulas import GaussianCopula, ClaytonCopula
from risksimpro.volatility_models import GARCH
from risksimpro.bayesian_calibration import BayesianCalibration
from risksimpro.variance_reduction_advanced import AdaptiveImportanceSampling
from risksimpro.realtime_ai import IncrementalUpdater, AISimulationOptimizer
from risksimpro.quantum_integration import QuantumMonteCarlo, HybridQuantumOrchestrator
from risksimpro.credit_risk_advanced import CreditDefaultModel, ExposureProfile, CVACalculator
from risksimpro.regulatory_compliance_advanced import BaselIIIReport
from risksimpro.visualization_dashboard import VisualizationDashboard

class TestDistributions(unittest.TestCase):
    def test_normal(self):
        dist = Normal(0, 1)
        samples = dist.sample(1000)
        self.assertAlmostEqual(np.mean(samples), 0, delta=0.1)
        self.assertAlmostEqual(np.std(samples), 1, delta=0.1)

    def test_student_t(self):
        dist = StudentT(5, 0, 1)
        samples = dist.sample(1000)
        self.assertEqual(len(samples), 1000)

    def test_mixture(self):
        components = [Normal(0,1), Normal(5,2)]
        weights = [0.3, 0.7]
        mixture = Mixture(components, weights)
        samples = mixture.sample(500)
        self.assertEqual(len(samples), 500)

class TestCopulas(unittest.TestCase):
    def test_gaussian_copula_fit_sample(self):
        u = np.random.uniform(0,1,(100,2))
        copula = GaussianCopula()
        copula.fit(u)
        samples = copula.sample(100)
        self.assertEqual(samples.shape, (100,2))

    def test_clayton_copula_sample(self):
        copula = ClaytonCopula(theta=2)
        samples = copula.sample(100, 2)
        self.assertEqual(samples.shape, (100,2))

class TestVolatilityModels(unittest.TestCase):
    def test_garch_fit_simulate(self):
        returns = np.random.normal(0,1,1000)
        model = GARCH()
        fit_result = model.fit(returns)
        self.assertIsNotNone(fit_result)
        sim_vols = model.simulate(10)
        self.assertEqual(len(sim_vols), 10)

class TestBayesianCalibration(unittest.TestCase):
    def test_calibration(self):
        observed = np.random.normal(0,1,100)
        def model_func():
            import pymc3 as pm
            mu = pm.Normal('mu', 0, 10)
            sigma = pm.HalfNormal('sigma', 10)
            return {'mu': mu, 'sigma': sigma}
        calib = BayesianCalibration(model_func, observed, draws=500, tune=500, chains=1)
        calib.calibrate()
        self.assertIsNotNone(calib.trace)

class TestVarianceReduction(unittest.TestCase):
    def test_adaptive_importance_sampling(self):
        target_pdf = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)
        ais = AdaptiveImportanceSampling(target_pdf)
        samples, weights = ais.run(num_samples=1000)
        self.assertEqual(len(samples), 1000)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=2)

class TestRealtimeAI(unittest.TestCase):
    def test_incremental_updater(self):
        updater = IncrementalUpdater()
        X1 = np.random.rand(10,3)
        y1 = np.random.rand(10)
        updater.update(X1, y1)
        preds = updater.predict(X1)
        self.assertEqual(len(preds), 10)

    def test_ai_simulation_optimizer(self):
        optimizer = AISimulationOptimizer(budget=1000)
        features = np.random.rand(5,2)
        alloc = optimizer.allocate(features)
        self.assertEqual(len(alloc), 5)
        self.assertAlmostEqual(np.sum(alloc), 1000)

class TestQuantumIntegration(unittest.TestCase):
    def test_quantum_mc(self):
        qm = QuantumMonteCarlo()
        qc = QuantumCircuit(1)
        qc.h(0)
        result = qm.run(qc, 0)
        self.assertTrue(0 <= result <= 1)

class TestCreditRisk(unittest.TestCase):
    def test_credit_default_simulation(self):
        prob = [0.01, 0.02, 0.03]
        model = CreditDefaultModel(prob)
        defaults = model.simulate_defaults(100)
        self.assertEqual(defaults.shape, (100, 3))

    def test_cva_calculator(self):
        cdm = CreditDefaultModel([0.01, 0.02])
        exposures = ExposureProfile([100, 200])
        cva = CVACalculator(cdm, exposures)
        val = cva.calculate_cva(50, 1.0, 5)
        self.assertIsInstance(val, float)

class TestRegulatoryCompliance(unittest.TestCase):
    def test_basel_report_generation(self):
        port_data = {'asset_count': 10}
        cap_req = {'CET1': 5, 'Tier1': 6, 'TotalCapital': 8}
        risk_metrics = {'risk_weighted_assets': 100}
        report = BaselIIIReport(port_data, cap_req, risk_metrics)
        result = report.generate_report()
        self.assertIn('compliance_status', result)

class TestVisualizationDashboard(unittest.TestCase):
    def test_dashboard_init(self):
        import pandas as pd
        data = pd.DataFrame({'scenario': [1,1,2,2], 'loss': [0.1, 0.2, 0.3, 0.4], 'pnl':[0.05,0.1,0.15,0.2]})
        dashboard = VisualizationDashboard(data)
        # We test the setup does not throw errors
        self.assertIsNotNone(dashboard.app)

if __name__ == "__main__":
    unittest.main()
