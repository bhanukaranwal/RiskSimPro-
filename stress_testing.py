class StressTestManager:
    def __init__(self):
        self.stress_tests = []

    def add_stress_test(self, stress_test):
        self.stress_tests.append(stress_test)

    def remove_stress_test(self, stress_test):
        self.stress_tests.remove(stress_test)

    def run_all(self, portfolio):
        results = {}
        for test in self.stress_tests:
            results[test.name] = test.apply(portfolio)
        return results
