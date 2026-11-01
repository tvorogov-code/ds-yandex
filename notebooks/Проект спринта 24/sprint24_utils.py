from pyspark import SparkConf, StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

MAX_AUDITION_HOURS = 24
MAX_CONTENT_LENGTH_HOURS = 24

# Создание Spark-сессии с параметрами в виде словаря
def create_spark_session_dict(appName, master="yarn", conf_dict=None):
    if conf_dict is None:
        conf_dict = {}
        
    if isinstance(conf_dict, dict):
        conf_pairs = conf_dict.items()
    else:
        conf_pairs = conf_dict

    return (
        SparkSession.builder
        .appName(appName)
        .config(conf=SparkConf().setAll(conf_pairs))
        .master(master)
        .getOrCreate()
    )

# Фильтрация данных по временному интервалу и геопозиции
def filter_data(df, start_date=None, end_date=None, usage_geo_id=None):
    if start_date is not None:
        df = df.filter(F.col('msk_business_dt') >= start_date)
        
    if end_date is not None:
        df = df.filter(F.col('msk_business_dt') <= end_date)
        
    if usage_geo_id is not None:
        df = df.filter(F.col('usage_geo_id') == usage_geo_id)
        
    return df

# Предобработка данных
def preprocess_data(df_audition, df_content):
    df_audition = df_audition.filter((F.col('hours') > 0) &
                                     (F.col('hours') <= MAX_AUDITION_HOURS))
    df_content = df_content.filter((F.col('main_content_duration_hours') > 0) &
                                   (F.col('main_content_duration_hours') <= MAX_CONTENT_LENGTH_HOURS))
    df_audition = (df_audition
                   .withColumn('msk_business_dt', F.to_date(F.col('msk_business_dt_str'), 'yyyy-MM-dd'))
                   .fillna({'app_version': '0'}))

    return df_audition, df_content

# Разделение списка тэгов тем на отдельные строки датасета
def split_published_topics(df):
    return (df.withColumn('published_topic_title', F.explode(F.split(F.col('published_topic_title_list'), ',\\s*'))))

# Построение витрин данных
def build_marts(df_audition, df_content, start_date=None, end_date=None, geo_id=None, repartition_cols = None):
    df_audition, df_content = preprocess_data(df_audition, df_content)
    df_audition = filter_data(df_audition, start_date, end_date, geo_id)

    df_joined = df_audition.join(df_content, on='main_content_id', how='left')

    # Разделение по указанным столбцам
    if repartition_cols:
        part_cols = [col.strip() for col in repartition_cols.split(',') if col.strip() in df_joined.columns]
        if part_cols:
            df_joined = df_joined.repartition(*part_cols)

    df_joined_filtered_split = split_published_topics(df_joined)

    user_content_mart_full = (df_joined
                        .groupBy('puid', 'main_content_id', 
                                'adult_content_flg', 
                                'kids_content_flg', 'main_content_type',
                                'main_content_duration_hours')
                        .agg(
                            F.sum('hours').alias('total_hours'),
                            F.count('audition_id').alias('sessions_count'),
                            F.min('msk_business_dt').alias('first_listening_date'),
                            F.max('msk_business_dt').alias('last_listening_date'),
                            F.mean('hours').alias('avg_session_hours')
                            )
                        .withColumn('finished_percent',
                                    F.when(F.col('main_content_duration_hours') == 0, 100)
                                    .otherwise(F.least(F.lit(1.0), F.col('total_hours') / F.col('main_content_duration_hours')) * 100))
                        .withColumn('is_completed', F.col('finished_percent') >= 100)
                        .withColumn('days_spent', F.datediff(F.col('last_listening_date'), F.col('first_listening_date')) + 1))

    user_content_mart_full.repartition('puid').persist(StorageLevel.MEMORY_AND_DISK)
    
    user_content_mart = (user_content_mart_full
                        .drop('first_listening_date', 'main_content_duration_hours', 
                            'adult_content_flg', 'kids_content_flg', 'main_content_type'))
    
    user_genre_content_mart = (df_joined_filtered_split
                    .groupBy('puid', 
                                'published_topic_title', 
                                'main_content_id', 
                                'main_content_duration_hours', 
                                'adult_content_flg', 
                                'kids_content_flg')
                    .agg(
                        F.sum('hours').alias('total_hours_cont'),
                        F.count('audition_id').alias('sessions_count_cont')
                        ))

    # Логика поиска любимого жанра, платформы, типа контента
    favourites_window = (Window.partitionBy('puid')
                            .orderBy(F.col('total_hours').desc(),
                                    F.col('sessions_count').desc()))

    user_genre_mart = (user_genre_content_mart
                    .groupBy('puid', 'published_topic_title')
                    .agg(
                        F.sum('total_hours_cont').alias('total_hours'),
                        F.sum('sessions_count_cont').alias('sessions_count'),
                        F.count_if(F.col('adult_content_flg')).alias('adult_count'),
                        F.count_if(F.col('kids_content_flg')).alias('kids_count'),
                        F.count_if(F.col('total_hours_cont') >= F.col('main_content_duration_hours'))
                        .alias('completed_count')
                        )
                        .withColumnRenamed('published_topic_title', 'genre')
                        .withColumn('genre_rank', 
                                    F.dense_rank()
                                    .over(favourites_window)))

    user_genre_mart.repartition('puid').persist(StorageLevel.MEMORY_AND_DISK)
    
    # Агрегация по жанру
    user_genre_agg = (user_genre_mart
                    .groupBy('puid')
                    .agg(F.count('genre').alias('unique_genres_count'),
                        F.min_by('genre', 'genre_rank').alias('favourite_genre')))

    # Агрегация по платформе
    user_platform_agg = (df_audition
                        .groupBy('puid', 'usage_platform_ru')
                        .agg(
                            F.count('audition_id').alias('sessions_count'),
                            F.sum('hours').alias('total_hours')
                            )
                        .withColumn('rank', F.dense_rank().over(favourites_window))
                        .filter(F.col('rank') == 1)
                        .select('puid', F.col('usage_platform_ru').alias('favourite_platform')))

    # Итоговая витрина
    user_mart = (user_content_mart_full
                .join(user_genre_agg, on='puid', how='inner')
                .join(user_platform_agg, on='puid', how='inner')
                .withColumn('favorite_content_type', F.first('main_content_type').over(favourites_window))
                .groupBy('puid', 'unique_genres_count', 'favourite_genre', 'favorite_content_type', 'favourite_platform')
                .agg(
                    F.sum('total_hours').alias('total_hours'),
                    F.count('main_content_id').alias('unique_content_count'),
                    F.min('first_listening_date').alias('first_listening_date'),
                    F.max('last_listening_date').alias('last_date'),
                    (100 * F.mean(F.col('is_completed').cast('int'))).alias('completion_rate'),
                    F.count_if('adult_content_flg').alias('adult_content_count'),
                    F.count_if('kids_content_flg').alias('kids_content_count')
                    )
                .withColumn('active_days', F.datediff(F.col('last_date'), F.col('first_listening_date')) + 1)
                .drop('first_listening_date', 'last_listening_date'))
        
    content_platform_agg = (df_audition
                        .groupBy('main_content_id')
                        .agg(F.countDistinct('usage_platform_ru').alias('platforms_count')))

    content_mart = (user_content_mart_full
                    .groupBy('main_content_id', 'main_content_duration_hours')
                    .agg(
                        F.count('puid').alias('unique_users'),
                        F.sum('total_hours').alias('total_hours'),
                        F.mean(F.when(F.col('main_content_duration_hours') == 0, 100)
                                .otherwise(100 * F.least(F.lit(1), F.col('total_hours') / F.col('main_content_duration_hours'))))
                        .alias('avg_finished_percent')
                        )
                    .join(content_platform_agg, on='main_content_id', how='inner')
                    .join(df_audition.agg(F.countDistinct('puid').alias('total_users')), how='cross')
                    .withColumn('popularity', F.col('unique_users') / F.col('total_users'))
                    .drop('total_users', 'main_content_duration_hours'))

    daily_user_mart = (df_audition
                    .groupBy('puid', F.col('msk_business_dt').alias('session_date'))
                    .agg(
                        F.countDistinct('main_content_id').alias('content_count'),
                        F.sum('hours').alias('hours_spent'),
                        F.countDistinct('audition_id').alias('sessions_count')
                    )
                    .withColumn('day_of_week', F.dayofweek('session_date')))

    return {"user_content_mart": user_content_mart, 
            "user_genre_mart": user_genre_mart, 
            "user_mart": user_mart, 
            "content_mart": content_mart, 
            "daily_user_mart": daily_user_mart}


