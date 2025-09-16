import pexpect
import time
import json
from base64 import b64decode, b64encode
import os
import sys
import time
import zlib
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "../../")))


IMPORT_TIMEOUT = 100
# PROOF_TIMEOUT = 120
PROOF_TIMEOUT = int(os.environ.get("PROOF_TIMEOUT", 300))

HOME_DIR = os.path.expanduser("~")

DEFAULT_LAKE_PATH = f"{HOME_DIR}/.elan/bin/lake"


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
    child: pexpect.spawn,
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


def lean4worker(
    proof_code_dict,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
    imports=DEFAULT_IMPORTS,
):
    """Worker function that continuously picks tasks and executes them."""
    child, _ = initiate_child()  # Start Lean 4 REPL

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

        response = send_command_and_wait(
            child,
            proof_code,
            env=0,
            allTactics=allTactics,
            ast=ast,
            premises=premises,
            tactics=tactics,
            imports=imports,
        )  # Run proof

        response["name"] = proof_name

        response["verify_time"] = round(time.time() - start_time, 2)

    try:
        child.close()
    except Exception:
        child.terminate(force=True)
    return response


def run(data: str):
    json_str = zlib.decompress(b64decode(data), 16 + zlib.MAX_WBITS).decode("utf-8")
    task_content = json.loads(json_str)
    ret = lean4worker(task_content)
    ret_str = json.dumps(ret)
    compressed = zlib.compress(
        ret_str.encode("utf-8"), level=9, wbits=16 + zlib.MAX_WBITS
    )
    compressed_b64 = b64encode(compressed).decode("utf-8")
    print(compressed_b64)


def main():
    inp = sys.argv[1]
    # inp = json.dumps({"cmd": "def f := 2"})
    # compressed = zlib.compress(inp.encode("utf-8"), level=9, wbits=16 + zlib.MAX_WBITS)
    # compressed_b64 = b64encode(compressed).decode("utf-8")
    run(inp)


if __name__ == "__main__":
    main()
