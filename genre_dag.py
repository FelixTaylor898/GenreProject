# scripts/genre_dag.py
"""
Airflow DAG for the Automated Book Genre Classification pipeline.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Local imports for pipeline tasks
from batch_ingest import ingest_genre_data
from transform import clean_genre_data
from train_model import train_genre_model

# --- Default settings for all tasks ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 10, 28),
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# --- Define the DAG itself ---
with DAG(
    dag_id="genre_project_dag",
    default_args=default_args,
    description="Daily ETL pipeline for Automated Book Genre Classification",
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # Step 1: Ingest the raw dataset from S3
    ingest_task = PythonOperator(
        task_id="ingest_raw_books",
        python_callable=ingest_genre_data,
    )

    # Step 2: Clean and normalize the dataset
    clean_task = PythonOperator(
        task_id="clean_book_data",
        python_callable=clean_genre_data,
    )

    # Step 3: Train and evaluate the ML model
    train_task = PythonOperator(
        task_id="train_genre_model",
        python_callable=train_genre_model,
    )

    # Define task dependencies (pipeline order)
    ingest_task >> clean_task >> train_task