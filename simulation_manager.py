import concurrent.futures
import numpy as np

class SimulationManager:
    def __init__(self, simulation_engine, max_workers=None):
        self.simulation_engine = simulation_engine
        self.max_workers = max_workers

    def run_parallel(self, portfolio, num_simulations, num_workers=None):
        if num_workers is None:
            num_workers = self.max_workers

        simulations_per_worker = num_simulations // num_workers
        futures = []
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                futures.append(executor.submit(self.simulation_engine.simulate, portfolio))

            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        combined_results = np.concatenate(results)
        return combined_results
