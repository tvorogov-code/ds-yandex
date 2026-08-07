import logging
from io import StringIO

import pandas as pd

from sqlalchemy import text

from airflow.providers.postgres.hooks.postgres import PostgresHook

DB_CONNECTION_ID = 'db_sprint19'

logger = logging.getLogger(__name__)

def _get_db_engine():
    hook = PostgresHook(postgres_conn_id=DB_CONNECTION_ID)
    return hook.get_conn()

def load_all_data():
    db_eng = _get_db_engine()

    df_events = pd.read_sql_table('events', con=db_eng)
    df_sessions = pd.read_sql_table('sessions', con=db_eng)
    df_orders = pd.read_sql_table('orders', con=db_eng)
    df_customers = pd.read_sql_table('customers', con=db_eng)

    return {"df_events": df_events, 
            "df_sessions": df_sessions, 
            "df_orders": df_orders, 
            "df_customers": df_customers}

def load_data_by_run_date(run_date_s: str):
    run_date = pd.Timestamp(run_date_s)
    db_eng = _get_db_engine()

    evt_query = """
        SELECT *
        FROM public.events
        WHERE DATE_TRUNC('day', timestamp) BETWEEN %(run_date)s - 30 AND %(run_date)s - 1
    """
    session_query = """
        SELECT *
        FROM public.sessions
        WHERE DATE_TRUNC('day', start_time) BETWEEN %(run_date)s - 30 AND %(run_date)s - 1
    """
    orders_query = """
        SELECT *
        FROM public.orders
        WHERE DATE_TRUNC('day', order_time) BETWEEN %(run_date)s - 30 AND %(run_date)s - 1
    """
    customers_query = """
        SELECT *
        FROM public.customers
        WHERE DATE_TRUNC('day', signup_date) < %(run_date)s - 30
    """

    df_events = pd.read_sql(evt_query, con=db_eng , params={"run_date": run_date.date()})
    df_sessions = pd.read_sql(session_query, con=db_eng, params={"run_date": run_date.date()})
    df_orders = pd.read_sql(orders_query, con=db_eng, params={"run_date": run_date.date()})
    df_customers = pd.read_sql(customers_query, con=db_eng, params={"run_date": run_date.date()})

    logger.info("Загружены данные за период с %s по %s", run_date - pd.Timedelta(days=30), 
                run_date - pd.Timedelta(days=1))

    return {"df_events": df_events, 
            "df_sessions": df_sessions, 
            "df_orders": df_orders, 
            "df_customers": df_customers}
