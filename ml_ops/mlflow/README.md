# Custom MLflow Server Docker image

## Building
Change the active directory to `/ml_ops/mlflow`.

```sh
export MLFLOW_VERSION=1.14.1
docker buildx build --platform=linux/arm64 --tag 573518775438.dkr.ecr.us-east-2.amazonaws.com/mlflow:$MLFLOW_VERSION .

```
