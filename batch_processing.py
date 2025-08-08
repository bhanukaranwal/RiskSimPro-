import queue
import threading

class BatchJobQueue:
    def __init__(self):
        self.jobs = queue.Queue()
        self.results = []
        self.lock = threading.Lock()

    def add_job(self, job):
        self.jobs.put(job)

    def worker(self):
        while not self.jobs.empty():
            job = self.jobs.get()
            result = job()
            with self.lock:
                self.results.append(result)
            self.jobs.task_done()

    def run(self, num_workers=4):
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self.results

class DistributedJobManager:
    def __init__(self, cluster_client):
        self.cluster_client = cluster_client

    def submit_jobs(self, jobs):
        futures = [self.cluster_client.submit(job) for job in jobs]
        results = [future.result() for future in futures]
        return results
