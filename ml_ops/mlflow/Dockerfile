FROM python:3.8.0

WORKDIR /mlflow

COPY requirements.txt requirements.txt

RUN python -m pip install -U pip
RUN pip install -r requirements.txt

ARG PORT=${PORT:-80} \
    ARTIFACTS_BUCKET_NAME=${ARTIFACTS_BUCKET_NAME:-htd-smart-mirror-mlflow-bucket} \
    DB_HOST=${DB_HOST:-db} \
    DB_PORT=${DB_PORT:-5432} \
    DB_USERNAME=${DB_USERNAME:-mlflow_user} \
    DB_PASSWORD=${DB_PASSWORD:-mlflow} \
    DB_NAME=${DB_NAME:-mlflow_db}

ENV PORT=${PORT} \
    ARTIFACTS_BUCKET_NAME=${ARTIFACTS_BUCKET_NAME} \
    DB_HOST=${DB_HOST} \
    DB_PORT=${DB_PORT} \
    DB_USERNAME=${DB_USERNAME} \
    DB_PASSWORD=${DB_PASSWORD} \
    DB_NAME=${DB_NAME}

EXPOSE ${PORT}

ENTRYPOINT ["sh", "-c", "mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --default-artifact-root s3://${ARTIFACTS_BUCKET_NAME} \
    --backend-store-uri postgresql://${DB_USERNAME}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME} \
"]
