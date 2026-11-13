"""Database connection and engine configuration using SQLAlchemy 2.0."""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def get_db_url() -> str:
    """Constructs the PostgreSQL connection URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "olist_user")
    password = os.getenv("POSTGRES_PASSWORD", "olist_secret_pw")
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "olist_ecommerce")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine() -> Engine:
    """Creates and returns a SQLAlchemy Engine instance."""
    db_url = get_db_url()
    return create_engine(db_url, pool_pre_ping=True, echo=False)