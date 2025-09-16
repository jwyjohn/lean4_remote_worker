import requests
import time
import uuid
import json
import random
from datetime import datetime
from lean4run import lean4worker
import traceback
from Crypto.Cipher import AES
from base64 import b64decode, b64encode

aead_key = bytes.fromhex("9fb28c1637a6f4939c8c50269085cfa6")


def aead_dec(key: bytes, encrypted_data: dict) -> dict:
    """Decrypt data that was encrypted using AES-GCM."""
    nonce = b64decode(encrypted_data["nonce"])
    header = b64decode(encrypted_data["header"])
    ciphertext = b64decode(encrypted_data["ciphertext"])
    tag = b64decode(encrypted_data["tag"])

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(header)
    try:
        decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
        return json.loads(decrypted_data.decode("utf-8"))
    except Exception as e:
        traceback.print_exc()
        print(f"Decryption failed: {str(e)}")
        raise


def aead_enc(key: bytes, msg: dict):
    cipher = AES.new(key, AES.MODE_GCM)
    header = b"terry_cscs"
    data = json.dumps(msg).encode("utf-8")
    cipher.update(header)
    ciphertext, tag = cipher.encrypt_and_digest(data)

    json_k = ["nonce", "header", "ciphertext", "tag"]
    json_v = [
        b64encode(x).decode("utf-8") for x in (cipher.nonce, header, ciphertext, tag)
    ]
    ret = dict(zip(json_k, json_v))
    return ret


def enc(msg: dict):
    return aead_enc(aead_key, msg)


def dec(encrypted_data: dict):
    return aead_dec(aead_key, encrypted_data)


class TaskWorker:
    def __init__(self, worker_id=None, base_url="http://localhost:5001"):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.base_url = base_url
        self.processed_count = 0

    def get_task(self):
        """Get next task from queue"""
        url = f"{self.base_url}/task/get"
        response = requests.post(url, json=enc({"worker_id": self.worker_id}))

        if response.status_code == 200:
            task_data = response.json()
            return dec(task_data)
        elif response.status_code == 204:
            return None  # No tasks available
        else:
            raise Exception(f"Failed to get task: {response.text}")

    def submit_result(self, task_id: str, result: dict):
        """Submit task result"""
        url = f"{self.base_url}/task/submit"
        assert isinstance(result, dict)

        payload = {
            "worker_id": self.worker_id,
            "task_id": task_id,
            "task_result": result,
        }

        response = requests.post(url, json=enc(payload))

        if response.status_code == 200:
            return dec(response.json())["success"]
        else:
            raise Exception(f"Failed to submit result: {response.text}")

    def process_task(self, task_data):
        """Process a task - this is where the actual work happens"""
        task_content = task_data["task_content"]

        print(f"Processing task {task_data['task_id'][:8]}...")

        try:
            result = self.process_lean4_task(task_content)
            return result
        except Exception as e:
            return {
                "system_errors": f"UNEXPECTED ERROR: {e}",
                "trace": traceback.format_exc(),
            }

    def process_lean4_task(self, content: dict):
        """Process JSON-formatted task"""
        assert isinstance(content, dict)
        ret = lean4worker(self.worker_id, content)
        return ret

    def run(self, max_tasks=None, poll_interval=2):
        """Main worker loop"""
        print(f"Starting worker: {self.worker_id}")
        print(f"Server: {self.base_url}")
        print("-" * 50)

        while True:
            try:
                # Get next task
                task = self.get_task()

                if not task:
                    # print(f"No tasks available. Waiting {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue

                print(f"Got task: {task['task_id']}")
                print(f"Created: {task['creation_time']}")

                # Process the task
                result = self.process_task(task)

                # Submit result
                success = self.submit_result(task["task_id"], result)

                if success:
                    self.processed_count += 1
                    print(f"✓ Task completed successfully")
                    print(f"  Result: {result}")
                    print(f"  Total processed: {self.processed_count}")
                else:
                    print("✗ Failed to submit result")

                print("-" * 30)

                # Check if we've reached max tasks
                if max_tasks and self.processed_count >= max_tasks:
                    print(f"Reached maximum tasks ({max_tasks}). Stopping.")
                    break

            except KeyboardInterrupt:
                print("\nWorker stopped by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Waiting {poll_interval}s before retry...")
                time.sleep(poll_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task Queue Worker")
    parser.add_argument("--worker-id", help="Worker ID (default: auto-generated)")
    parser.add_argument("--server", default="http://localhost:5001", help="Server URL")
    parser.add_argument("--max-tasks", type=int, help="Maximum tasks to process")
    parser.add_argument(
        "--poll-interval", type=float, default=2, help="Polling interval in seconds"
    )

    args = parser.parse_args()

    worker = TaskWorker(worker_id=args.worker_id, base_url=args.server)

    try:
        worker.run(max_tasks=args.max_tasks, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("\nShutting down worker...")


if __name__ == "__main__":
    main()
