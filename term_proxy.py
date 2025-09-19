import multiprocessing as mp
import time
import traceback
import uuid
import json
from typing import Dict, Any
import os
from pwn import *
from base64 import b64decode, b64encode
import zlib
import requests
from Crypto.Cipher import AES
import numpy as np
import random
from subprocess import check_output

# Constants from second file
IMPORT_TIMEOUT = 100
PROOF_TIMEOUT = int(os.environ.get("PROOF_TIMEOUT", 300))
DEFAULT_IMPORTS = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

# Encryption key from second file
aead_key = bytes.fromhex("9fb28c1637a6f4939c8c50269085cfa6")


def encode_proof_code_dict(proof_code_dict: dict):
    ret_str = json.dumps(proof_code_dict)
    compressed = zlib.compress(
        ret_str.encode("utf-8"), level=9, wbits=16 + zlib.MAX_WBITS
    )
    compressed_b64 = b64encode(compressed).decode("utf-8")
    return compressed_b64


def decode_proof_code_dict(compressed_b64: str):
    data = zlib.decompress(b64decode(compressed_b64), 16 + zlib.MAX_WBITS).decode(
        "utf-8"
    )
    task_content: dict = json.loads(data)
    return task_content


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


def lean4worker(
    worker_id,
    proof_code_dict,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
    imports=DEFAULT_IMPORTS,
):
    """Worker function that executes Lean4 proofs via SSH."""
    print(f"Worker {worker_id} started Lean REPL.", flush=True)

    start_time = time.time()
    proof_code = proof_code_dict["code"]
    proof_name = proof_code_dict["name"]

    if len(proof_code) == 0:
        response = {
            "code": proof_code,
            "compilation_result": {
                "pass": False,
                "complete": False,
                "system_errors": None,
            },
        }
        response["name"] = proof_name
        response["verify_time"] = round(time.time() - start_time, 2)
    else:
        print(encode_proof_code_dict(proof_code_dict))
        response = check_output(
            [
                f"/cluster/home/wenyjiang/lean4workers/lean4_remote_worker/.venv/bin/python "
                f"/cluster/home/wenyjiang/lean4workers/lean4_remote_worker/lean4_run_b64.py "
                f"{encode_proof_code_dict(proof_code_dict)}",
            ],
            timeout=PROOF_TIMEOUT,
        )
        response = response.decode("utf-8").strip()
        response = decode_proof_code_dict(response)

    print(f"Worker {worker_id} terminated Lean REPL.", flush=True)
    return response


def http_post_request_local(url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Print request to stdout and wait for response from stdin (for remote server)"""
    try:
        request = {"url": url, "json": json_data}

        # Print encoded request to stdout
        encoded_request = encode_proof_code_dict(request)
        print(f"HTTP_REQUEST:{encoded_request}", flush=True)

        # Wait for encoded response from stdin
        response_line = input()
        response_data = decode_proof_code_dict(
            response_line.removeprefix("HTTP_RESPONSE:")
        )

        return response_data

    except Exception as e:
        return {"status_code": 500, "error": str(e)}


def server_process_local(request_queue: mp.Queue, result_queues: Dict[str, mp.Queue]):
    """Server process that handles HTTP requests (runs on remote machine)"""
    print(f"Server started (PID: {os.getpid()})")

    while True:
        try:
            request_data = request_queue.get()
            if request_data is None:  # Shutdown signal
                break

            client_id, request = request_data

            # Handle HTTP POST request
            url = request["url"]
            json_data = request["json"]

            response = http_post_request_local(url, json_data)
            result_queues[client_id].put(response)

        except KeyboardInterrupt:
            break

    print("Server shutting down")


class TaskWorkerSSH:
    def __init__(
        self,
        worker_id: str,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        base_url="http://f00d1ab.ddns.net:5001",
    ):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.base_url = base_url
        self.processed_count = 0

    def http_post(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP POST request via server and wait for response (blocking)"""
        request = {"url": url, "json": json_data}

        self.request_queue.put((self.worker_id, request))
        response = self.result_queue.get()  # Block until response
        return response

    def get_task(self) -> Dict[str, Any]:
        """Get next task from remote queue"""
        url = f"{self.base_url}/task/get"
        encrypted_payload = enc({"worker_id": self.worker_id})

        response = self.http_post(url, encrypted_payload)

        if response.get("status_code") == 200:
            # Decrypt the response data
            decrypted_data = dec(response["data"])
            return {"status_code": 200, "data": decrypted_data}
        else:
            return response

    def submit_result(self, task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Submit task result to remote server"""
        url = f"{self.base_url}/task/submit"

        payload = {
            "worker_id": self.worker_id,
            "task_id": task_id,
            "task_result": result,
        }
        encrypted_payload = enc(payload)

        response = self.http_post(url, encrypted_payload)

        if response.get("status_code") == 200:
            decrypted_data = dec(response["data"])
            return {"status_code": 200, "success": decrypted_data["success"]}
        else:
            return response

    def process_task(self, task_data):
        """Process a task - this is where the actual work happens"""
        task_content = task_data["task_content"]

        print(f"Processing task {task_data['task_id'][:8]}...", flush=True)

        try:
            result = self.process_lean4_task(task_content)
            return result
        except Exception as e:
            return {
                "system_errors": f"UNEXPECTED ERROR: {e}",
                "trace": traceback.format_exc(),
            }

    def process_lean4_task(self, content: dict):
        """Process JSON-formatted Lean4 task"""
        assert isinstance(content, dict)
        ret = lean4worker(self.worker_id, content)
        return ret

    def run(self, max_tasks: int = None, poll_interval: int = 2):
        """Main worker loop"""
        print(f"TaskWorker {self.worker_id} started (PID: {os.getpid()})", flush=True)

        while True:
            try:
                # Get next task from remote server
                # print(f"TaskWorker {self.worker_id} getting task...")
                task_response = self.get_task()

                if task_response.get("status_code") == 204:
                    print(
                        f"TaskWorker {self.worker_id}: No tasks available. Waiting {poll_interval}s...",
                        flush=True,
                    )
                    time.sleep(poll_interval)
                    continue
                elif task_response.get("status_code") != 200:
                    print(
                        f"TaskWorker {self.worker_id}: Error getting task: {task_response}",
                        flush=True,
                    )
                    time.sleep(poll_interval)
                    continue

                # Extract task from response
                remote_task = task_response["data"]
                task_id = remote_task["task_id"]

                print(f"TaskWorker {self.worker_id}: Got task {task_id}", flush=True)
                print(
                    f"TaskWorker {self.worker_id}: Created: {remote_task.get('creation_time', 'unknown')}",
                    flush=True,
                )

                # Process the task
                result = self.process_task(remote_task)

                print(f"TaskWorker {self.worker_id}: Task completed", flush=True)

                # Submit result back to remote server
                submit_response = self.submit_result(task_id, result)

                if submit_response.get("status_code") == 200 and submit_response.get(
                    "success"
                ):
                    self.processed_count += 1
                    print(
                        f"TaskWorker {self.worker_id}: ✓ Result submitted successfully",
                        flush=True,
                    )
                    print(
                        f"TaskWorker {self.worker_id}: Result: {json.dumps(result)}",
                        flush=True,
                    )
                    print(
                        f"TaskWorker {self.worker_id}: Total processed: {self.processed_count}",
                        flush=True,
                    )
                else:
                    print(
                        f"TaskWorker {self.worker_id}: ✗ Failed to submit result: {submit_response}",
                        flush=True,
                    )

                print("-" * 30, flush=True)

                # Check if we've reached max tasks
                if max_tasks and self.processed_count >= max_tasks:
                    print(
                        f"TaskWorker {self.worker_id}: Reached maximum tasks ({max_tasks}). Stopping.",
                        flush=True,
                    )
                    break

            except KeyboardInterrupt:
                print(f"\nTaskWorker {self.worker_id} stopped by user", flush=True)
                break
            except Exception as e:
                print(f"TaskWorker {self.worker_id}: Error: {e}", flush=True)
                traceback.print_exc()
                print(
                    f"TaskWorker {self.worker_id}: Waiting {poll_interval}s before retry...",
                    flush=True,
                )
                time.sleep(poll_interval)

        print(f"TaskWorker {self.worker_id} finished", flush=True)


def main_remote():
    """Main function that runs on remote machine"""
    num_workers = 3

    # Generate UUID for each worker
    worker_ids = [str(uuid.uuid4()) for _ in range(num_workers)]

    # Create shared queues
    request_queue = mp.Queue()
    result_queues = {worker_id: mp.Queue() for worker_id in worker_ids}

    # Start server process
    server = mp.Process(
        target=server_process_local, args=(request_queue, result_queues)
    )
    server.start()

    # Start TaskWorker processes
    workers = []
    for worker_id in worker_ids:
        worker_process = mp.Process(
            target=lambda wid=worker_id: TaskWorkerSSH(
                wid, request_queue, result_queues[wid]
            ).run(max_tasks=2)
        )
        worker_process.start()
        workers.append(worker_process)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Send shutdown signal to server and wait
    request_queue.put(None)
    server.join()

    print("All processes finished")


class HTTPProxyServer:
    def __init__(self):
        self.script_path = __file__
        self.ssh_process = None
        self.is_running = False

    def start_remote_process(self):
        """SSH to remote machine and start the main process"""
        try:
            print(f"Connecting to remote...")

            self.ssh_process = process(["ssh", "euler-tunnel"])

            # Transfer the script content
            script_content = open(__file__, "r").read()

            # Create a temporary script on remote machine
            remote_script = f"/tmp/merged_worker_{uuid.uuid4().hex[:8]}.py"

            self.ssh_process.sendline(f"cat > {remote_script} << 'EOF'")
            self.ssh_process.sendline(script_content)
            self.ssh_process.sendline("EOF")

            # Wait for file transfer to complete
            time.sleep(1)

            # Run the script
            cmd = f"cd /tmp && /cluster/home/wenyjiang/lean4workers/lean4_remote_worker/.venv/bin/python {remote_script} remote"
            self.ssh_process.sendline(cmd)

            # Wait for the remote process to start
            time.sleep(2)

            print("Remote process started successfully")
            self.is_running = True

        except Exception as e:
            print(f"Failed to start remote process: {e}")
            traceback.print_exc()
            raise

    def handle_http_request(
        self, url: str, json_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle HTTP request by making actual HTTP call"""
        try:
            print(f"Proxy making HTTP POST to {url}")

            response = requests.post(url, json=json_data, timeout=30)
            if response.status_code == 200:
                return {"status_code": response.status_code, "data": response.json()}
            elif response.status_code == 204:
                return {"status_code": response.status_code}
            else:
                assert f"HTTP {response.status_code}" == ""

        except Exception as e:
            print(f"HTTP request failed: {e}")
            traceback.print_exc()
            return {"status_code": 500, "error": str(e)}

    def process_requests(self):
        """Main loop to process HTTP requests from remote machine"""
        print("HTTP Proxy started, waiting for requests...")

        while self.is_running:
            try:
                # Wait for HTTP request from remote machine
                line: str = self.ssh_process.recvline(timeout=2.0).decode().strip()

                if line.startswith("HTTP_REQUEST:"):
                    # Extract and decode the request (FIXED: use correct prefix)
                    encoded_request = line.removeprefix("HTTP_REQUEST:")
                    request_data = decode_proof_code_dict(encoded_request)

                    url = request_data["url"]
                    json_data = request_data["json"]

                    print(f"Received request: {url}")

                    # Make the actual HTTP request
                    response = self.handle_http_request(url, json_data)

                    # Encode and send response back
                    encoded_response = encode_proof_code_dict(response)
                    self.ssh_process.sendline(f"HTTP_RESPONSE:{encoded_response}")

                    print(f"Sent response: status_code={response.get('status_code')}")

                elif "All processes finished" in line:
                    print("Remote processes completed")
                    break
                elif line.strip():  # Print non-empty lines for debugging
                    print(f"Remote: {line}")

            except Exception as e:
                if "timeout" not in str(e).lower():
                    print(f"Error processing request: {e}")
                continue

    def run(self):
        """Start the proxy server"""
        try:
            self.start_remote_process()
            self.process_requests()

        except KeyboardInterrupt:
            print("\nProxy server interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up SSH connection"""
        self.is_running = False
        if self.ssh_process:
            try:
                self.ssh_process.close()
                print("SSH connection closed")
            except:
                pass


def main():
    """Main function - runs the HTTP proxy"""
    import argparse

    parser = argparse.ArgumentParser(description="Merged Task Worker with SSH Proxy")
    # parser.add_argument("--remote-host", default="192.168.1.78", help="Remote host")
    # parser.add_argument("--remote-user", default="user", help="Remote username")

    args = parser.parse_args()

    proxy = HTTPProxyServer()

    proxy.run()


if __name__ == "__main__":
    import sys

    # Check if we're running the remote version
    if len(sys.argv) > 1 and sys.argv[1] == "remote":
        main_remote()
    else:
        context.log_level = "DEBUG"
        main()
