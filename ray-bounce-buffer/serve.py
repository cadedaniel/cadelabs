#!/usr/bin/env python3

import requests
import sys
import time
from ray import serve
from ray.serve.drivers import DAGDriver
from ray.serve.dag import InputNode
from ray.serve.http_adapters import json_request


@serve.deployment
class DeployedModel:
    def __init__(self):
        pass

    def get_time(self, _):
        return time.time()

    def get_duration(self, start_time):
        return time.time() - start_time


def main():
    with InputNode() as input_node:
        model_1 = DeployedModel.bind()
        model_2 = DeployedModel.bind()
        dag = model_2.get_duration.bind(model_1.get_time.bind(input_node))

    serve.run(DAGDriver.bind(dag, http_adapter=json_request), port=1025)

    def measure():
        return requests.post("http://localhost:1025/", json="").json()

    durations_ms = sorted(
        [(i, 1000 * measure()) for i in range(15)], key=lambda x: x[1]
    )
    durations_fmt = [f"{i:03}: {duration:0.2f}" for i, duration in durations_ms]
    print("\n".join(durations_fmt))


if __name__ == "__main__":
    sys.exit(main())
