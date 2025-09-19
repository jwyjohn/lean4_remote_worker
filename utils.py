from base64 import b64decode, b64encode
import json
import zlib


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
