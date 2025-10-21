import os, httpx

# Default to local Node server (no Docker)
HCS_LOGGER_URL = os.getenv("HCS_LOGGER_URL", "http://127.0.0.1:3001/hcs/log")

async def log_to_hcs(message: str, memo: str = "", reference_id: str = "", hash_hex: str = ""):
    payload = {"message": message, "memo": memo, "referenceId": reference_id, "hash": hash_hex}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HCS_LOGGER_URL, json=payload)
        r.raise_for_status()
        return r.json()
