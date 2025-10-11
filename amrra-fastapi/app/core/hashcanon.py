import json, hashlib

def _sort_keys(obj):
    if isinstance(obj, dict):
        # omit 'report_hash' during hashing
        return {k: _sort_keys(v) for k,v in sorted(obj.items()) if k != "report_hash"}
    if isinstance(obj, list):
        return [ _sort_keys(x) for x in obj ]
    return obj

def canonical_string(obj) -> str:
    return json.dumps(_sort_keys(obj), separators=(",", ":"), ensure_ascii=False)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
