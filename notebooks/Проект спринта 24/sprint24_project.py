import os
from dotenv import load_dotenv
import argparse

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.logger import PySparkLogger

from sprint24_utils import create_spark_session_dict, build_marts

AUDITION_FILE_NAME = 'audition.parquet'
CONTENT_FILE_NAME = 'content.parquet'

def split_published_topics(df):
    return (df.withColumn('published_topic_title', F.explode(F.split(F.col('published_topic_title_list'), ',\\s*'))))

def main():

    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Проект спринта 24. Пайплайн построения витрин для сервиса книжных рекомендаций.")
    parser.add_argument("--start_date", required=False, help="Начальная дата периода")
    parser.add_argument("--end_date", required=False, help="Конечная дата периода")
    parser.add_argument("--geo_id", required=False, help="Идентификатор геолокации")
    parser.add_argument("--columns", required=False, help="Колонки для партицирования")
    parser.add_argument("--audition_path", required=True, help="Путь к файлу с данными о сессиях")
    parser.add_argument("--content_path", required=True, help="Путь к файлу с данными о контенте")
    parser.add_argument("--output_path", required=True, help="Путь для сохранения результата")

    args = parser.parse_args()
    
    load_dotenv()

    logger = PySparkLogger.getLogger()
    spark = create_spark_session_dict(appName="Sprint_24_project",
                                     master="yarn",
                                     conf_dict={
                                        "spark.executor.memory": "2g",
                                        "spark.executor.cores": "2",
                                        "spark.executor.instances": "2",
                                        "spark.driver.cores": "4",
                                        "spark.driver.memory": "8g",
                                        "spark.sql.shuffle.partitions": "32",
                                        "spark.network.timeout": "300",
                                        "spark.jars.packages": "org.apache.spark:spark-hadoop-cloud_2.13:3.5.0,software.amazon.awssdk:bundle:2.35.4",
                                        "spark.hadoop.fs.s3a.access.key": os.getenv("S3_ACCESS_KEY"),
                                        "spark.hadoop.fs.s3a.secret.key": os.getenv("S3_SECRET_KEY"),
                                        "spark.hadoop.fs.s3a.endpoint": os.getenv("S3_ENDPOINT"),
                                        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                                        "spark.hadoop.fs.s3a.path.style.access": "true"
                                    })

    df_audition = None
    try:    
        df_audition = spark.read.parquet(args.audition_path)
        df_content = spark.read.parquet(args.content_path)

        start_date = args.start_date
        end_date = args.end_date
        geo_id = args.geo_id
        columns = args.columns
        output_path = args.output_path

        marts = build_marts(df_audition, 
                            df_content, 
                            start_date=start_date, 
                            end_date=end_date, 
                            geo_id=geo_id, 
                            repartition_cols=columns)
        
        for name, mart in marts.items():
            logger.warn(f"Запись витрины {name} в {output_path}/{name}")
            mart.write.mode('overwrite').parquet(f"{output_path}/{name}")
            logger.warn(f"Витрина записана")

        logger.warn("План построения витрины")
        marts['user_mart'].explain(mode='extended')

    except Exception as e:
        print(f"Ошибка: {e}")
        raise e

    finally:
        spark.stop()

if __name__ == "__main__":
    main()