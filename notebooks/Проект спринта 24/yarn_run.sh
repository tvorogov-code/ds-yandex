#!/bin/bash

spark-submit --master yarn \
            --deploy-mode cluster \
            --files .env \
            --py-files sprint24_utils.py \
            sprint24_project.py \
                --start_date=2024-01-01 \
                --end_date=2025-12-31 \
                --geo_id="Москва, Россия" \
                --columns=puid,genre \
                --audition_path=s3a://s3-ds-source/audition.parquet \
                --content_path=s3a://s3-ds-source/content.parquet \
                --output_path=s3a://s3-ds-20260728-6ce7e7aa05/results/S24_project_data
