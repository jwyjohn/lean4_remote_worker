#!/usr/bin/env python3

import multiprocessing as mp
import signal
import sys
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WorkerInfo:
    process: mp.Process
    start_time: datetime
    restart_count: int = 0


class WorkerDaemon:
    def __init__(
        self,
        num_workers: int = 4,
        base_url: str = "http://localhost:5001",
        max_restarts: int = 5,
        restart_delay: float = 2.0,
        max_tasks_per_worker: Optional[int] = None,
        poll_interval: int = 2,
    ):

        self.num_workers = num_workers
        self.base_url = base_url
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.max_tasks_per_worker = max_tasks_per_worker
        self.poll_interval = poll_interval

        self.workers: Dict[str, WorkerInfo] = {}
        self.shutdown_requested = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("worker_daemon.log"),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    def _worker_target(self, worker_id: str):
        """Target function for worker processes"""
        try:
            from lean4_worker import TaskWorker  # Replace with actual import

            worker = TaskWorker(worker_id=worker_id, base_url=self.base_url)

            worker.run(
                max_tasks=self.max_tasks_per_worker, poll_interval=self.poll_interval
            )

        except Exception as e:
            logging.error(f"Worker {worker_id} failed: {e}")
            raise

    def _start_worker(self, worker_id: str) -> mp.Process:
        """Start a single worker process"""
        process = mp.Process(
            target=self._worker_target, args=(worker_id,), name=f"Worker-{worker_id}"
        )
        process.start()
        self.logger.info(f"Started worker {worker_id} (PID: {process.pid})")
        return process

    def _should_restart_worker(self, worker_id: str, worker_info: WorkerInfo) -> bool:
        """Determine if a worker should be restarted"""
        if worker_info.restart_count >= self.max_restarts:
            self.logger.error(
                f"Worker {worker_id} exceeded max restarts ({self.max_restarts})"
            )
            return False
        return True

    def _restart_worker(self, worker_id: str):
        """Restart a failed worker"""
        old_info = self.workers[worker_id]

        if not self._should_restart_worker(worker_id, old_info):
            return False

        self.logger.info(
            f"Restarting worker {worker_id} (attempt {old_info.restart_count + 1})"
        )

        # Wait before restart to avoid rapid restart loops
        time.sleep(self.restart_delay)

        # Start new process
        new_process = self._start_worker(worker_id)

        # Update worker info
        self.workers[worker_id] = WorkerInfo(
            process=new_process,
            start_time=datetime.now(),
            restart_count=old_info.restart_count + 1,
        )

        return True

    def start_workers(self):
        """Start all worker processes"""
        self.logger.info(f"Starting {self.num_workers} workers...")

        for i in range(self.num_workers):
            worker_id = f"daemon-worker-{i:02d}"
            process = self._start_worker(worker_id)

            self.workers[worker_id] = WorkerInfo(
                process=process, start_time=datetime.now()
            )

    def monitor_workers(self):
        """Monitor worker processes and restart failed ones"""
        while not self.shutdown_requested:
            dead_workers = []

            for worker_id, worker_info in self.workers.items():
                if not worker_info.process.is_alive():
                    exit_code = worker_info.process.exitcode
                    uptime = datetime.now() - worker_info.start_time

                    self.logger.warning(
                        f"Worker {worker_id} died (exit code: {exit_code}, "
                        f"uptime: {uptime}, restarts: {worker_info.restart_count})"
                    )

                    dead_workers.append(worker_id)

            # Restart dead workers
            for worker_id in dead_workers:
                if self.shutdown_requested:
                    break

                if not self._restart_worker(worker_id):
                    # Remove worker that can't be restarted
                    del self.workers[worker_id]
                    self.logger.error(f"Permanently removing worker {worker_id}")

            # Check if we have any workers left
            if not self.workers:
                self.logger.error("All workers have failed and cannot be restarted")
                break

            time.sleep(1)  # Check every second

    def shutdown_workers(self):
        """Gracefully shutdown all workers"""
        self.logger.info("Shutting down workers...")

        # Send SIGTERM to all workers
        for worker_id, worker_info in self.workers.items():
            if worker_info.process.is_alive():
                self.logger.info(f"Terminating worker {worker_id}")
                worker_info.process.terminate()

        # Wait for graceful shutdown
        shutdown_timeout = 10
        start_time = time.time()

        while time.time() - start_time < shutdown_timeout:
            alive_workers = [
                worker_id
                for worker_id, worker_info in self.workers.items()
                if worker_info.process.is_alive()
            ]

            if not alive_workers:
                break

            time.sleep(0.5)

        # Force kill any remaining workers
        for worker_id, worker_info in self.workers.items():
            if worker_info.process.is_alive():
                self.logger.warning(f"Force killing worker {worker_id}")
                worker_info.process.kill()
                worker_info.process.join(timeout=2)

    def run(self):
        """Main daemon loop"""
        try:
            self.logger.info("Worker daemon starting...")
            self.logger.info(
                f"Configuration: {self.num_workers} workers, "
                f"max restarts: {self.max_restarts}, "
                f"restart delay: {self.restart_delay}s"
            )

            self.start_workers()
            self.monitor_workers()

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Daemon error: {e}")
        finally:
            self.shutdown_workers()
            self.logger.info("Worker daemon stopped")

    def status(self):
        """Print current status of all workers"""
        print(f"Worker Daemon Status ({len(self.workers)} workers)")
        print("-" * 60)

        for worker_id, worker_info in self.workers.items():
            status = "ALIVE" if worker_info.process.is_alive() else "DEAD"
            uptime = datetime.now() - worker_info.start_time

            print(
                f"{worker_id:20} | {status:5} | "
                f"PID: {worker_info.process.pid:6} | "
                f"Restarts: {worker_info.restart_count:2} | "
                f"Uptime: {str(uptime).split('.')[0]}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task Worker Daemon")
    parser.add_argument(
        "--workers", type=int, default=96, help="Number of worker processes"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:5001", help="Base URL for task server"
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=300,
        help="Maximum restart attempts per worker",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=2.0,
        help="Delay between restart attempts",
    )
    parser.add_argument(
        "--max-tasks", type=int, help="Maximum tasks per worker before restart"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=2, help="Task polling interval in seconds"
    )

    args = parser.parse_args()

    daemon = WorkerDaemon(
        num_workers=args.workers,
        base_url=args.base_url,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        max_tasks_per_worker=args.max_tasks,
        poll_interval=args.poll_interval,
    )

    daemon.run()


if __name__ == "__main__":
    main()
