# app/core/hashcanon.py
import json

def canonical_string(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def sha256_hex(s: str) -> str:
    import hashlib
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()