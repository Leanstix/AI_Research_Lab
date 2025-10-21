# worker/worker.py
import os
from rq import Worker, Queue, Connection
from redis import Redis

listen = ['amrra']
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work(with_scheduler=False)