# AMRRA / back_end (skeleton)

A minimal back-end aligned with the AMRRA architecture, with:
- **FastAPI** entrypoint
- **Redis RQ** job queue and **worker** process (jobs run off-thread like the original intent)
- **Dynamic retrieval with web hydration** into `CORPUS_DIR`
- Reproducible **artifacts**: JSON (canonical-hashed) and PDF

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
redis-server   # or `docker compose up redis`
uvicorn app.api:app --reload
# in another terminal:
python worker/worker.py
```

## Docker compose
```bash
docker compose up --build
```

## API
- `POST /api/run` body: `{"question": "...", "experiment": "ml|ttest|literature"}`
- `GET  /api/runs/{run_id}` -> status + artifact links
- `GET  /api/artifacts/{filename}` -> download pdf/json

Jobs are executed by the separate worker process via Redis RQ.
```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"question":"Effects of regular sexual intercourse on prostate cancer risk"}'
```

## Env
- `CORPUS_DIR=./corpus` (hydrated web docs saved here)
- `ENABLE_LLM=true` + keys to enable LLM summaries/conclusions