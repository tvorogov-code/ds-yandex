from datetime import datetime, timedelta
import logging

import pandas as pd

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, get_current_context
from airflow.utils.task_group import TaskGroup
from airflow.providers.postgres.operators.postgres import PostgresOperator

from sprint19.db_utils import load_data_by_run_date
from sprint19.s3_utils import write_dfs_to_s3, read_dfs_from_s3, write_df_to_s3
from sprint19.batch_utils import preprocess, get_slices, create_batch_features

RUN_DATE_VAR = "run_date"

S3_RAW_DATA_PREFIX = "raw_data/"
S3_SLICES_PREFIX = "slices/"
S3_BATCH_FEATURES_PREFIX = "batch_features/"

default_args = {
    "owner": "tvorogov_as",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

logger = logging.getLogger(__name__)

# Получить результат предыдущей задачи из XCom
def get_prev_task_xcom(**context):
    task = context["task"]
    first_upstream = next(iter(task.upstream_task_ids), None)
    if first_upstream:
        return context["ti"].xcom_pull(task_ids=first_upstream)
    else:
        return None

# Получить параметр run_date из контекста вызова DAG
def get_run_date(**context):
    conf = context["dag_run"].conf
    run_date_str = conf.get(RUN_DATE_VAR) if conf else None
    if not run_date_str:
        run_date_str = f"{datetime.now().date():%Y-%m-%d}"
    return run_date_str

def load_data(**context):
    run_date = get_run_date(**context)
    data_dict = load_data_by_run_date(run_date)
    return write_dfs_to_s3(data_dict, f"{S3_RAW_DATA_PREFIX}{run_date}/")

def preprocess_and_slice_data(**context):
    run_date = get_run_date(**context)
    s3_keys = get_prev_task_xcom(**context)
    df_dict = read_dfs_from_s3(f"{S3_RAW_DATA_PREFIX}{run_date}/", s3_keys)
    df_preprocessed = preprocess(**df_dict)
    slices = get_slices(run_date, **df_preprocessed)
    return write_dfs_to_s3(slices, f"{S3_SLICES_PREFIX}{run_date}/")

def create_and_upload_batch_features(**context):
    run_date = get_run_date(**context)
    s3_keys = get_prev_task_xcom(**context)
    df_dict = read_dfs_from_s3(f"{S3_SLICES_PREFIX}{run_date}/", s3_keys)
    batch_features = create_batch_features(run_date, df_dict)
    return write_df_to_s3(batch_features, f"{S3_BATCH_FEATURES_PREFIX}{run_date}.parquet")

with DAG(
    dag_id="calculate_batch_features",
    default_args=default_args,
    description="Вычисление batch-признаков для срезов данных и их загрузка в S3",
    schedule_interval=timedelta(days=1), 
    start_date=datetime(2024, 1, 31),
    catchup=False,
    tags=["ml", "batch", "daily"],
    max_active_runs=1,
    params={"run_date": "2025-05-01"}
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        provide_context=True
    )

    preprocess_and_slice_task = PythonOperator(
        task_id="preprocess_and_slice_data",
        python_callable=preprocess_and_slice_data,
        provide_context=True
    )

    create_batch_features_task = PythonOperator(
        task_id="create_and_upload_batch_features",
        python_callable=create_and_upload_batch_features,
        provide_context=True
    )

    load_data_task >> preprocess_and_slice_task >> create_batch_features_task