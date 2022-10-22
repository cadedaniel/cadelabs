#!/usr/bin/env python

import ray

runtime_env = {"pip": ["jax[cpu]",]}
ray.init(runtime_env=runtime_env)


@ray.remote
class JaxWorker:
    def __init__(self):
        pass

    def do_shit(self):
        import socket
        return socket.gethostname()


def install_jax_on_all_worker_nodes():
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    num_nodes = 4

    pg = placement_group(
        bundles=[dict(CPU=1) for _ in range(num_nodes)],
        strategy="STRICT_SPREAD",
    )
    ray.get(pg.ready())

    # TODO get cluster ips
    workers = [
        JaxWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=rank
            )
        ).remote()
        for rank in range(4)
    ]
    hostnames = ray.get([w.do_shit.remote() for w in workers])
    assert len(set(hostnames)) == len(hostnames), f"Hostnames not unique {hostnames}"


@ray.remote
def task():
    import jax

    print(jax)

install_jax_on_all_worker_nodes()
