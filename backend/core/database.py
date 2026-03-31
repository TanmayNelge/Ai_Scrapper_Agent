"""
Database models with SQLAlchemy.
SQLite in WAL mode for concurrent reads during crawl.
"""
import json
import time
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, Boolean,
    ForeignKey, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from ..config import CONFIG

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    query = Column(String(500), nullable=False)
    mode = Column(String(20), default="TEXT_ONLY")  # TEXT_ONLY or SINGLE_IMAGE
    schema_json = Column(Text, nullable=False)       # JSON string of schema dict
    max_depth = Column(Integer, default=3)
    status = Column(String(20), default="idle")      # idle, running, completed, failed
    created_at = Column(Float, default=time.time)
    total_items = Column(Integer, default=0)

    results = relationship("Result", back_populates="project", cascade="all, delete-orphan")

    @property
    def schema_dict(self) -> dict:
        return json.loads(self.schema_json) if self.schema_json else {}


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    data_json = Column(Text, nullable=False)
    source_url = Column(String(2000))
    created_at = Column(Float, default=time.time)

    project = relationship("Project", back_populates="results")

    @property
    def data(self) -> dict:
        return json.loads(self.data_json) if self.data_json else {}


class EventLog(Base):
    """Persistent event log for WebSocket reconnection."""
    __tablename__ = "event_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    event_type = Column(String(20), nullable=False)
    payload = Column(Text)
    url = Column(String(2000))
    created_at = Column(Float, default=time.time)


# ─── Engine + Session setup ──────────────────────────────────────

def create_db_engine(db_url: str = None):
    url = db_url or CONFIG.db_url
    engine = create_engine(url, echo=False, connect_args={"check_same_thread": False})

    # Enable WAL mode for concurrent reads
    @event.listens_for(engine, "connect")
    def set_sqlite_wal(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def init_db(engine=None):
    if engine is None:
        engine = create_db_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None):
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(bind=engine)
