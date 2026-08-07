from io import BytesIO
import logging
from typing import Tuple, Optional

import pandas as pd

import boto3
from botocore.exceptions import ClientError
from airflow.hooks.base import BaseHook

logger = logging.getLogger(__name__)

CONNECTION_ID = 's3_sprint19'

def _get_s3_client_and_bucket(conn_id: str) -> Tuple[object, str]:
    conn = BaseHook.get_connection(conn_id)
    extra = conn.extra_dejson or {}

    endpoint_url = extra.get("endpoint_url")
    bucket = extra.get("bucket")

    if not endpoint_url or not bucket:
        raise ValueError(
            "В Airflow Connection (Extra) должны быть endpoint_url и bucket. "
            "Пример Extra: {'endpoint_url': 'https://storage.yandexcloud.net', 'bucket': 'my-bucket'}"
        )

    s3 = boto3.client(
        "s3",
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        endpoint_url=endpoint_url,
    )
    return s3, bucket

def upload_bytes_to_s3(data: bytes, s3_key: str, conn_id: str, bucket: Optional[str] = None) -> str:
    """
    Загрузка bytes в S3 по ключу.
    Возвращает s3_key (для XCom).
    """
    s3, bucket_from_conn = _get_s3_client_and_bucket(conn_id)
    bucket = bucket or bucket_from_conn

    logger.info("Загрузка на S3: s3://%s/%s (size=%s bytes)", bucket, s3_key, len(data))

    s3.put_object(Bucket=bucket, Key=s3_key, Body=data)

    logger.info("✓ Uploaded: s3://%s/%s", bucket, s3_key)
    return s3_key

def write_df_to_s3(df, s3_key):
    p_buffer = df.to_parquet(index=False, engine="pyarrow")
    upload_bytes_to_s3(p_buffer, s3_key, CONNECTION_ID)
    return s3_key

def write_dfs_to_s3(df_dict, s3_key_prefix):
    s3_keys = []
    for key, df in df_dict.items():
        s3_key = f"{s3_key_prefix}{key}.parquet"
        write_df_to_s3(df, s3_key)
        s3_keys.append(key)
    return s3_keys

def read_dfs_from_s3(s3_key_prefix, s3_keys):
    s3, bucket = _get_s3_client_and_bucket(CONNECTION_ID)
    df_dict = {}
    for s3_key in s3_keys:
        logger.info("Чтение из S3: s3://%s/%s", bucket, s3_key_prefix + s3_key + ".parquet")
        obj = s3.get_object(Bucket=bucket, Key=s3_key_prefix + s3_key + ".parquet")
        buffer = BytesIO(obj["Body"].read())
        df = pd.read_parquet(buffer, engine="pyarrow")
        df.info()
        df_dict[s3_key] = df
    return df_dict