import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

celery_app = Celery(
    "amrra",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
)
