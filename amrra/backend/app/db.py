import os
import pathlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/amrra.db")

# If using a local SQLite path like sqlite:///./data/amrra.db, make sure folder exists
if DATABASE_URL.startswith("sqlite:///"):
    path = DATABASE_URL.replace("sqlite:///", "", 1)
    folder = pathlib.Path(path).parent
    folder.mkdir(parents=True, exist_ok=True)

# SQLite needs check_same_thread=False for typical FastAPI/Celery usage
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        future=True,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
