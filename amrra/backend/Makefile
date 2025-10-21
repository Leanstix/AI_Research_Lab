.PHONY: venv api worker node

venv:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

api:
	. .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	. .venv/bin/activate && celery -A app.celery_app.celery_app worker --loglevel=INFO --concurrency=2

node:
	cd scripts/node-hcs-server && npm i && cp -n .env.sample .env || true && npm run start
