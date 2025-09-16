import requests
import time
import uuid
import json
from datetime import datetime
import traceback
from Crypto.Cipher import AES
from base64 import b64decode, b64encode
import os
import sys
import time
import traceback
import pexpect
import multiprocessing as mp
import random
import numpy as np
import zlib
from subprocess import check_output


def split_list_randomly(lst, k):
    random.shuffle(lst)  # Shuffle the list randomly
    return list(
        map(list, np.array_split(lst, k))
    )  # Split into k approximately equal parts


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "../../")))


IMPORT_TIMEOUT = 100
# PROOF_TIMEOUT = 120
PROOF_TIMEOUT = int(os.environ.get("PROOF_TIMEOUT", 300))

HOME_DIR = os.path.expanduser("~")

DEFAULT_LAKE_PATH = f"{HOME_DIR}/.elan/bin/lake"


# DEFAULT_LEAN_WORKSPACE="mathlib4/"
DEFAULT_LEAN_WORKSPACE = "/home/jwyjohn/Services/lab-lean4tasks/mathlib4/"


DEFAULT_IMPORTS = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def initiate_child(imports=DEFAULT_IMPORTS):
    # Start the Lean 4 REPL using pexpect
    # Note: Adjust the command if necessary for your setup
    # child = pexpect.spawn('stty -icanon', cwd=lean_workspace, encoding='utf-8', maxread=1, echo=False)

    child = pexpect.spawn(
        f"/bin/bash",
        cwd=DEFAULT_LEAN_WORKSPACE,
        encoding="utf-8",
        maxread=1,
        echo=False,
    )

    # # Uncomment the next line to see the REPL's output for debugging
    # child.logfile = sys.stdout

    child.sendline("stty -icanon")

    child.sendline(f"cd {DEFAULT_LEAN_WORKSPACE}")

    child.sendline(f"{DEFAULT_LAKE_PATH} exe repl")

    response = send_command_and_wait(child, imports, timeout=IMPORT_TIMEOUT)

    # print(f"Initializing Lean REPL: (PID: {child.pid})", flush = True)
    # return child

    return child, response


def send_command_and_wait(
    child,
    command,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
    env=None,
    timeout=PROOF_TIMEOUT,
    imports=DEFAULT_IMPORTS,
):
    """
    Send a JSON command to the Lean REPL and wait for the output.
    The REPL output is expected to be a JSON dict (possibly spanning multiple lines)
    ending with a double newline.
    """
    # Build the JSON command
    if env is None:
        json_cmd = json.dumps({"cmd": command})
    else:
        json_cmd = json.dumps(
            {
                "cmd": command,
                "allTactics": allTactics,
                "ast": ast,
                "premises": premises,
                "tactics": tactics,
                "env": env,
            }
        )

    child.sendline(json_cmd)
    child.sendline("")  # This sends the extra newline.
    print(json_cmd)

    # import pdb; pdb.set_trace()

    code = imports + command
    try:
        # Wait for the output delimiter (double newline)
        child.expect(["\r\n\r\n", "\n\n"], timeout=timeout)
        # pexpect.before contains everything up to the matched delimiter.
        response = child.before.strip()
        # print(response)

        block = response

        # problem_id = proof_code_list[i]["name"]
        try:
            result = json.loads(block)
            # ast_results = lean4_parser(command, result['ast']) if 'ast' in result and result['ast'] else {}
            ast_results = {}
            parsed_result = {
                "sorries": result.get("sorries", []),
                "tactics": result.get("tactics", []),
                "errors": [
                    m
                    for m in result.get("messages", [])
                    if m.get("severity") == "error"
                ],
                "warnings": [
                    m
                    for m in result.get("messages", [])
                    if m.get("severity") == "warning"
                ],
                "infos": [
                    m for m in result.get("messages", []) if m.get("severity") == "info"
                ],
                "ast": ast_results,
                # "verified_code": code,
                # "problem_id": problem_id
                "system_errors": None,
            }
            parsed_result["pass"] = not parsed_result["errors"]
            parsed_result["complete"] = (
                parsed_result["pass"]
                and not parsed_result["sorries"]
                and not any(
                    "declaration uses 'sorry'" in warning["data"]
                    or "failed" in warning["data"]
                    for warning in parsed_result["warnings"]
                )
            )

        except json.JSONDecodeError as e:
            import traceback

            traceback.print_exc()
            parsed_result = {
                "pass": False,
                "complete": False,
                # "verified_code": code,
                # "problem_id": problem_id,
                "system_errors": f"JSONDECODE ERROR: {e}",
            }

        response = {"code": command, "compilation_result": parsed_result}

    except pexpect.TIMEOUT as e:
        response = {
            "code": command,
            "compilation_result": {
                "pass": False,
                "complete": False,
                "system_errors": f"TIMEOUT ERROR: {e}",
            },
        }
    except pexpect.EOF as e:
        response = {
            "code": command,
            "compilation_result": {
                "pass": False,
                "complete": False,
                "system_errors": f"EOF ERROR: {e}",
            },
        }
    except Exception as e:  # Catch any other unexpected errors
        response = {
            "code": command,
            "compilation_result": {
                "pass": False,
                "complete": False,
                "system_errors": f"UNEXPECTED ERROR: {e}",
            },
        }
    return response


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
    task_content = json.loads(data)
    return task_content


def lean4worker(
    worker_id,
    proof_code_dict,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
    imports=DEFAULT_IMPORTS,
):
    """Worker function that continuously picks tasks and executes them."""
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

        start_time = time.time()

    else:
        print(encode_proof_code_dict(proof_code_dict))
        response = check_output(
            [
                "ssh",
                "euler-tunnel",
                "/cluster/home/wenyjiang/lean4workers/lean4_remote_worker/.venv/bin/python",
                "/cluster/home/wenyjiang/lean4workers/lean4_remote_worker/lean4_run_b64.py",
                encode_proof_code_dict(proof_code_dict),
            ],
            timeout=PROOF_TIMEOUT,
        )
        response = response.decode("utf-8")
        response = decode_proof_code_dict(response)

    print(f"Worker {worker_id} terminated Lean REPL.", flush=True)
    return response


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
    def __init__(self, worker_id=None, base_url="http://f00d1ab.ddns.net:5001"):
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
    parser.add_argument(
        "--server", default="http://f00d1ab.ddns.net:5001", help="Server URL"
    )
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
