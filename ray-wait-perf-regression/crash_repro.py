#!/usr/bin/env python3

import ray


def node_assignment(node):
    return {'node:{}'.format(node['NodeName']): 0.001}


@ray.remote
def source_task():
    print('src task')
    ds = ray.data.range_tensor(10)
    return ds


@ray.remote
def destination_task(other_node):
    future = source_task.options(
        resources=node_assignment(other_node)
    ).remote()

    print('dest task')
    ds = ray.get(future)
    print(ds.size_bytes())


if __name__ == '__main__':
    ray.init()

    # Need to guarantee that the src and dest tasks run on different nodes.
    nodes = ray.nodes()
    assert len(nodes) >= 2, "Must have at least two nodes to reproduce crash"
    destination_node = nodes[0]
    source_node = nodes[1]

    future = destination_task.options(
        resources=node_assignment(destination_node),
    ).remote(source_node)

    ray.get(future)
