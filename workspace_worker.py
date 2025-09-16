from base64 import b64decode, b64encode
import json
import os
from subprocess import check_output
import zlib
import srsly
import time
import random
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process
from glob import glob

PROOF_TIMEOUT = int(os.environ.get("PROOF_TIMEOUT", 300))

WORKSPACE_PATH = Path("workspace")
TODO_PATH = WORKSPACE_PATH / "todo/"
DONE_PATH = WORKSPACE_PATH / "done/"
GLOBAL_LOCK_PATH = WORKSPACE_PATH / "global.lock"


def random_delay():
    dt = random.randint(100, 500)
    time.sleep(dt / 200)


def is_g_locked():
    return GLOBAL_LOCK_PATH.exists()


def g_lock():
    while is_g_locked():
        random_delay()
    print("GLock")
    GLOBAL_LOCK_PATH.mkdir()


def g_unlock():
    random_delay()
    print("GUnLock")
    assert is_g_locked()
    GLOBAL_LOCK_PATH.rmdir()


def is_f_locked(path: str):
    return Path(path + ".lock").exists()


def f_lock(path: str):
    while is_f_locked(path):
        random_delay()
    Path(path + ".lock").mkdir()


def f_unlock(path: str):
    assert is_f_locked(path)
    Path(path + ".lock").rmdir()


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


def lean4worker(proof_code_dict: dict):
    """Worker function that continuously picks tasks and executes them."""

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
        # print(encode_proof_code_dict(proof_code_dict))
        response = check_output(
            [
                f"/home/jwyjohn/Services/lab-lean4tasks/lean4_remote_worker/.venv/bin/python",
                f"/home/jwyjohn/Services/lab-lean4tasks/lean4_remote_worker/lean4_run_b64.py",
                f"{encode_proof_code_dict(proof_code_dict)}",
            ],
            timeout=PROOF_TIMEOUT,
        )
        response = response.decode("utf-8").strip()
        response = decode_proof_code_dict(response)
    return response


# 1. Get global lock
# 2. Find a json file in TODO_PATH that is not locked by string order
# 3. Lock that json file
# 4. Release global lock
# 5. Call lean4worker, save the result as the same filename in DONE_PATH
# 6. Get global lock
# 7. Unlock that json file in TODO_PATH
# 8. Remove that json file in TODO_PATH
# 9. Release global lock
def worker_loop():
    # 1. Get global lock
    g_lock()

    try:
        # 2. Find a json file in TODO_PATH that is not locked by string order
        # print(glob("./workspace/todo/*.gz"))

        todo_files = sorted(
            [
                f
                for f in glob("./workspace/todo/*.gz")
                if f.endswith(".json.gz") and not is_f_locked(f)
            ]
        )
        if not todo_files:
            # No tasks available
            g_unlock()
            return

        task_file = todo_files[0]  # First by string order
        print(task_file)

        # 3. Lock that json file
        f_lock(task_file)

        # 4. Release global lock
        g_unlock()

        try:
            # Process the task
            proof_code_dict = srsly.read_gzip_json(task_file)
            print(proof_code_dict)

            # 5. Call lean4worker, save the result as the same filename in DONE_PATH
            result = lean4worker(proof_code_dict)
            print(result)

            done_file = "./workspace/done/" + task_file.split("/")[-1]
            print(done_file)
            srsly.write_gzip_json(done_file, result)

            # 6. Get global lock
            g_lock()

            try:
                # 7. Unlock that json file in TODO_PATH
                f_unlock(task_file)

                # 8. Remove that json file in TODO_PATH
                Path(task_file).unlink()

                # 9. Release global lock
                g_unlock()

            except Exception:
                raise

        except Exception:
            # Clean up file lock if processing failed
            if is_f_locked(task_file):
                try:
                    f_unlock(task_file)
                except Exception:
                    raise
            raise

    except Exception:
        import traceback

        traceback.print_exc()
        g_unlock()


def worker_process():
    """Wrapper function that runs the worker loop continuously."""
    while True:
        try:
            worker_loop()
            # Small delay to prevent busy waiting when no tasks available
            time.sleep(0.01)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(0.1)  # Brief pause before retrying


def run_workers(num_workers=64):
    """Start multiple worker processes."""
    processes = []

    try:
        # Create and start worker processes
        for i in range(num_workers):
            p = Process(target=worker_process, name=f"Worker-{i}")
            p.start()
            processes.append(p)

        print(f"Started {num_workers} worker processes")

        # Wait for all processes to complete (or until interrupted)
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\nShutting down workers...")

        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()

        # Wait for processes to terminate
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()  # Force kill if still alive

        print("All workers stopped")


if __name__ == "__main__":
    # Ensure workspace directories exist
    try:
        GLOBAL_LOCK_PATH.rmdir()
    except:
        pass
    WORKSPACE_PATH.mkdir(exist_ok=True)
    TODO_PATH.mkdir(exist_ok=True)
    DONE_PATH.mkdir(exist_ok=True)

    # Run 64 concurrent workers
    run_workers(3)
