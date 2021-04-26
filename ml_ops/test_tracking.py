import os
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts, set_tracking_uri

if __name__ == "__main__":
    print("Running mlflow_tracking.py")
    set_tracking_uri("http://localhost:5000")

    log_param("param1", randint(0, 100))

    log_metric("foo", random())
    log_metric("foo", random() + 2)
    log_metric("foo", random() + 4)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")
