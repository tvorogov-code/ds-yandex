import logging

import pandas as pd

S3_CONNECTION_ID = 's3_sprint19'
    
logger = logging.getLogger(__name__)

# Предобработка загруженных данных
def preprocess(df_events, df_sessions, df_orders, df_customers):
    df_customers['signup_date'] = pd.to_datetime(df_customers['signup_date'])

    return {"df_events": df_events, 
            "df_sessions": df_sessions, 
            "df_orders": df_orders, 
            "df_customers": df_customers}

# Получение 7- и 30-дневных срезов
def get_slices(run_date_s, df_events, df_sessions, df_orders, df_customers):
    run_date = pd.Timestamp(run_date_s)
    df_dict = {}

    date_from_7 = run_date - pd.Timedelta(days=7)
    date_from_30 = run_date - pd.Timedelta(days=30)

    if df_events.empty:
        raise ValueError(f"Пустой df_events для даты {run_date_s}")
    if date_from_30 < df_events['timestamp'].min().normalize():
        raise ValueError(f"30-дневный срез для этой даты недоступен " + 
                         f"({date_from_30} < {df_events['timestamp'].min().normalize()})")

    # Фильтруем df_customers, чтобы оставить только тех клиентов, которые зарегистрировались 30 и более дней назад
    df_customers_ids = df_customers[df_customers['signup_date'] <= date_from_30][['customer_id']]

    df_dict['events_7'] = (df_events[(df_events['timestamp'] >= date_from_7) &
                                     (df_events['timestamp'] < run_date)]
                           .merge(df_sessions[['session_id', 'customer_id']], on='session_id')
                           .merge(df_customers_ids, on='customer_id', how='inner'))
    
    df_dict['events_30'] = (df_events[(df_events['timestamp'] >= date_from_30) &
                                      (df_events['timestamp'] < run_date)]
                           .merge(df_sessions[['session_id', 'customer_id']], on='session_id')
                           .merge(df_customers_ids, on='customer_id', how='inner'))

    df_dict['sessions_7'] = (df_sessions[(df_sessions['start_time'] >= date_from_7) &
                                        (df_sessions['start_time'] < run_date)]
                            .merge(df_customers_ids, on='customer_id', how='inner'))
    
    df_dict['sessions_30'] = (df_sessions[(df_sessions['start_time'] >= date_from_30) &
                                         (df_sessions['start_time'] < run_date)]
                            .merge(df_customers_ids, on='customer_id', how='inner'))
    
    df_dict['orders_7'] = (df_orders[(df_orders['order_time'] >= date_from_7) &
                                    (df_orders['order_time'] < run_date)]
                            .merge(df_customers_ids, on='customer_id', how='inner'))
    
    df_dict['orders_30'] = (df_orders[(df_orders['order_time'] >= date_from_30) &
                                     (df_orders['order_time'] < run_date)]
                            .merge(df_customers_ids, on='customer_id', how='inner'))

    for df in df_dict.values():
        if df.empty:
            raise ValueError(f"Пустой срез данных для даты {run_date_s}.")
    
    return df_dict

# Создание батч-признаков
def create_batch_features(run_date_s, df_dict):
    df = {}
    run_date = pd.Timestamp(run_date_s)

    df['page_views_7_count'] = (df_dict['events_7'][df_dict['events_7']['event_type'] == 'page_view']
                                .groupby('customer_id')
                                .size()
                                .rename('page_views_7_count'))

    df['page_views_30_count'] = (df_dict['events_30'][df_dict['events_30']['event_type'] == 'page_view']
                                .groupby('customer_id')
                                .size()
                                .rename('page_views_30_count'))

    df['add_to_cart_7_count'] = (df_dict['events_7'][df_dict['events_7']['event_type'] == 'add_to_cart']
                                .groupby('customer_id')
                                .size()
                                .rename('add_to_cart_7_count'))
    
    df['add_to_cart_30_count'] = (df_dict['events_30'][df_dict['events_30']['event_type'] == 'add_to_cart']
                                .groupby('customer_id')
                                .size()
                                .rename('add_to_cart_30_count'))

    df_purchase_count_7 = (df_dict['events_7'][df_dict['events_7']['event_type'] == 'purchase']
                                .groupby('customer_id')
                                .size()
                                .rename('purchase_count_7'))
        
    df_purchase_count_30 = (df_dict['events_30'][df_dict['events_30']['event_type'] == 'purchase']
                                .groupby('customer_id')
                                .size()
                                .rename('purchase_count_30'))

    df['cart_conv_7'] = (df['add_to_cart_7_count'] / df['page_views_7_count']).rename('cart_conv_7')
    df['cart_conv_30'] = (df['add_to_cart_30_count'] / df['page_views_30_count']).rename('cart_conv_30')
    df['purchase_conv_7'] = (df_purchase_count_7 / df['add_to_cart_7_count']).rename('purchase_conv_7')
    df['purchase_conv_30'] = (df_purchase_count_30 / df['add_to_cart_30_count']).rename('purchase_conv_30')
    df['unique_products_7'] = df_dict['events_7'].groupby('customer_id')['product_id'].nunique().rename('unique_products_7')
    df['unique_products_30'] = df_dict['events_30'].groupby('customer_id')['product_id'].nunique().rename('unique_products_30')

    df_session_durations = (df_dict['sessions_30']
                            .merge(df_dict['events_30']
                                      .groupby('session_id')['timestamp']
                                      .max()
                                      .rename('last_event_time'), 
                                    on='session_id', how='left'))
    df_session_durations['session_duration'] = (df_session_durations['last_event_time'] - 
                                                df_session_durations['start_time']).dt.total_seconds()
    df_session_durations['session_duration'] = df_session_durations['session_duration'].fillna(0)
    
    df['mean_session_duration_30'] = (df_session_durations
                                        .groupby('customer_id')['session_duration']
                                        .mean()
                                        .rename('mean_session_duration_30'))
    df['sessions_count_7'] = df_dict['sessions_7'].groupby('customer_id').size().rename('sessions_count_7')
    df['sessions_count_30'] = df_dict['sessions_30'].groupby('customer_id').size().rename('sessions_count_30')

    df_last_purchase = (
        df_dict['events_7']
        .loc[df_dict['events_7']['event_type'] == 'purchase']
        .groupby('customer_id')['timestamp']
        .max()
    )

    df['days_from_last_purchase'] = (
        (run_date.normalize() - df_last_purchase.dt.normalize()).dt.days + 1
    ).rename('days_from_last_purchase')

    df_dict['orders_30']['total_usd'] = df_dict['orders_30']['total_usd'].fillna(0)
    df['orders_count_30'] = df_dict['orders_30'].groupby('customer_id').size().rename('orders_count_30')
    df['total_usd_sum_30'] = df_dict['orders_30'].groupby('customer_id')['total_usd'].sum().rename('total_usd_sum_30')
    df['total_usd_mean_30'] = df_dict['orders_30'].groupby('customer_id')['total_usd'].mean().rename('total_usd_mean_30')

    # Объединение всех признаков в один DataFrame
    batch_features = pd.concat(df.values(), axis=1)

    # Заполнение пропущенных значений из-за деления на ноль или отсутствия покупок
    batch_features.fillna({'days_from_last_purchase': -1}, inplace=True)
    batch_features.fillna(0, inplace=True)

    batch_features['run_date'] = run_date
    
    return batch_features.reset_index()
