"""Script to ingest all 9 Olist Kaggle CSV files into PostgreSQL."""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.db_config import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mapping of Kaggle CSV filenames to clean PostgreSQL table names
CSV_TABLE_MAP: Dict[str, str] = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
}

# Columns that must be parsed as timestamps during ingestion
DATE_COLUMNS: Dict[str, List[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def create_indexes(engine: Engine) -> None:
    """Creates B-Tree indexes on primary/foreign keys to speed up SQL joins."""
    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders (customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (order_status);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items (order_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items (product_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON order_items (seller_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_payments_order_id ON order_payments (order_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_reviews_order_id ON order_reviews (order_id);",
        "CREATE INDEX IF NOT EXISTS idx_customers_zip ON customers (customer_zip_code_prefix);",
        "CREATE INDEX IF NOT EXISTS idx_sellers_zip ON sellers (seller_zip_code_prefix);",
        "CREATE INDEX IF NOT EXISTS idx_geo_zip ON geolocation (geolocation_zip_code_prefix);",
    ]
    with engine.begin() as conn:
        for query in index_queries:
            conn.execute(text(query))
    logger.info("Successfully created database indexes.")


def load_csvs_to_postgres(raw_data_dir: Path, engine: Engine) -> None:
    """Reads all 9 CSV files from raw_data_dir and loads them into PostgreSQL."""
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

    for csv_file, table_name in CSV_TABLE_MAP.items():
        file_path = raw_data_dir / csv_file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing '{csv_file}' in {raw_data_dir}. "
                "Please download the Olist dataset from Kaggle first."
            )

        logger.info("Reading %s...", csv_file)
        parse_dates = DATE_COLUMNS.get(table_name, False)
        df = pd.read_csv(file_path, parse_dates=parse_dates)

        logger.info(
            "Loading %d rows into table '%s'...", len(df), table_name
        )
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=10000,
            method="multi",
        )
        logger.info("Finished loading table '%s'.", table_name)

    create_indexes(engine)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    db_engine = get_engine()
    load_csvs_to_postgres(raw_data_dir=raw_dir, engine=db_engine)