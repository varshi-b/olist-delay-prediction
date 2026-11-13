"""Executes the analytical SQL queries in queries.sql and displays results."""

from pathlib import Path
from typing import Dict
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.db_config import get_engine


def parse_sql_file(sql_path: Path) -> Dict[str, str]:
    """Parses a SQL file split by '-- [QUERY_NAME]' markers into a dictionary."""
    content = sql_path.read_text(encoding="utf-8")
    blocks = content.split("-- [")
    queries: Dict[str, str] = {}

    for block in blocks[1:]:
        header, sql_body = block.split("]", 1)
        query_name = header.strip()
        # Remove empty lines and trailing semicolons for clean execution
        clean_sql = sql_body.strip()
        queries[query_name] = clean_sql

    return queries


def execute_all_queries(engine: Engine, sql_path: Path) -> None:
    """Runs each parsed SQL query and prints a formatted pandas DataFrame."""
    queries = parse_sql_file(sql_path)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    with engine.connect() as conn:
        for name, sql in queries.items():
            print("=" * 80)
            print(f"Executing: {name}")
            print("=" * 80)
            df = pd.read_sql_query(text(sql), con=conn)
            print(df.to_string(index=False))
            print("\n")


if __name__ == "__main__":
    sql_file_path = Path(__file__).resolve().parent / "queries.sql"
    db_engine = get_engine()
    execute_all_queries(engine=db_engine, sql_path=sql_file_path)