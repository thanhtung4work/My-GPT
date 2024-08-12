FROM python:3.9-slim

LABEL maintainer="tungt"
LABEL organization="mlopsvn"

WORKDIR /mlflow/

RUN apt-get update -y
RUN apt-get install -y iputils-ping
RUN pip install --no-cache-dir mlflow

EXPOSE 5000

# Ref: https://github.com/mlflow/mlflow/issues/6274#issuecomment-1196106372
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000