#!/usr/bin/env python3

import requests
import sys
from ray import serve
from ray.serve.drivers import DAGDriver
from ray.serve.dag import InputNode
from ray.serve.http_adapters import json_request

@serve.deployment
class DeployedModel:
    def __init__(self):
        pass

    def predict(self, input):
        return 0

def main():
    with InputNode() as input_node:
        model_1 = DeployedModel.bind()
        model_2 = DeployedModel.bind()
        dag = model_2.predict.bind(
            model_1.predict.bind(input_node)
        )

    serve.run(DAGDriver.bind(dag, http_adapter=json_request), port=1025)

    print(requests.post("http://localhost:1025/", json="hello world").json())

if __name__ == '__main__':
    sys.exit(main())
