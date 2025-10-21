# queue/broker.py
import os
from redis import Redis
from rq import Queue

def get_queue():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    conn = Redis.from_url(redis_url)
    return Queue("amrra", connection=conn)