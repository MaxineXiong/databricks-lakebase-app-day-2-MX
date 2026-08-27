"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password.
This keeps setup to a single secret instead of five separate env vars.
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor

_w = WorkspaceClient()

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


def _lakebase_url() -> str:
    """Fetch and decode the Lakebase connection URL from the Databricks secret scope."""
    secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase (lazy import avoids hard dependency)."""
    from sqlalchemy import create_engine
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


# DDL for weather data tables
DDL_WEATHER_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS public.weather_documents (
    id VARCHAR(255) PRIMARY KEY,
    location VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    headline TEXT NOT NULL,
    narrative_text TEXT,
    effective_at TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL
);
"""

DDL_WEATHER_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS public.weather_embeddings (
    id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

DDL_WEATHER_EMBEDDINGS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding_hnsw
ON public.weather_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""


def init_schema():
    """Initialize the weather data schema (creates tables if they don't exist)."""
    with get_connection() as conn:
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS public.weather_documents CASCADE")
            cur.execute("DROP TABLE IF EXISTS public.weather_embeddings")
            # Enable vector extension for pgvector support
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Create tables
            cur.execute(DDL_WEATHER_DOCUMENTS)
            cur.execute(DDL_WEATHER_EMBEDDINGS)
            # Create an index for faster vector search performance
            cur.execute(DDL_WEATHER_EMBEDDINGS_INDEX)
    print("Weather schema initialized successfully.")

