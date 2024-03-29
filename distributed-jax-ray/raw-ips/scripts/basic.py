#!/usr/bin/env python

import ray
import time
import os

#juntime_env = {"pip": ["jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",]}
#runtime_env = {}
runtime_env = {'working_dir': '.'}
ray.init(runtime_env=runtime_env)


@ray.remote(num_gpus=1)
class JaxWorker:
    def __init__(self, rank, world_size):
        try:
            import jax
        except ImportError:
            print('Jax not installed where JaxWorker is running')
            raise

        self.rank = rank
        self.world_size = world_size

    def get_hostname(self):
        import socket
        return socket.gethostname()

    def get_coordinator_address(self):
        from ray.train._internal.utils import get_address_and_port
        address, port = get_address_and_port()
        return f'{address}:{port}'
    
    def run_proc(self, command: str, coordinator_address: str, *, env_mixin: dict = None):
        env_mixin = env_mixin or {}

        print('run proc')

        env_mixin['COORDINATOR_ADDRESS'] = coordinator_address
        env_mixin['WORLD_SIZE'] = str(self.world_size)
        env_mixin['WORLD_RANK'] = str(self.rank)

        # TODO improve this when more accelerators
        #env_mixin['CUDA_VISIBLE_DEVICES'] = '0'
        env_mixin['NCCL_SOCKET_IFNAME'] = 'ens5'
        #env_mixin['NCCL_DEBUG'] = 'INFO'
        #env_mixin['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

        # TODO get coord address from rank 0
        run_background_job(command, env_mixin)

    def get_interfaces(self):
        import socket
        return socket.if_nameindex()

def create_actors_on_each_node():
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    world_size = 4

    from ray.serve._private.utils import get_current_node_resource_key
    bundles = [dict(CPU=1, GPU=1) for _ in range(world_size)]

    # Rank 0 on head node
    bundles[0][get_current_node_resource_key()] = 0.01

    print('Bundles:', bundles)

    pg = placement_group(
        bundles=bundles,
        strategy="STRICT_SPREAD",
    )
    ray.get(pg.ready())

    print('PG ready')

    # TODO get cluster ips
    workers = [
        JaxWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=rank
            ),
            runtime_env=runtime_env
        ).remote(rank, world_size)
        for rank in range(world_size)
    ]

    print('Getting hostnames')

    hostnames = ray.get([w.get_hostname.remote() for w in workers])
    if len(hostnames) == 4:
        assert len(set(hostnames)) == len(hostnames), f"Hostnames not unique {hostnames}"

    return workers

import typer

app = typer.Typer()
@app.command()
def spmd(command: str):
    workers = create_actors_on_each_node()

    print('Getting coordinator address')
    coordinator_address = ray.get(workers[0].get_coordinator_address.remote())
    print(f'Coordinator address: {coordinator_address}')

    #command = 'python jax-example.py'
    #command = 'nvidia-smi'
    ray.get([w.run_proc.remote(command, coordinator_address) for w in workers])

@app.command()
def interfaces(skip_loopback: bool = True):
    workers = create_actors_on_each_node()
    interfaces = ray.get([w.get_interfaces.remote() for w in workers])
    names = set(nic[1] for host in interfaces for nic in host)

    if skip_loopback:
        names = set(name for name in names if name != 'lo' and name != 'docker0')

    print(names)


@app.command()
def verify():
    create_actors_on_each_node()

import subprocess
def _run_kill_child(
    *popenargs, input=None, timeout=None, check=False, **kwargs
) -> subprocess.CompletedProcess:
    """
    This function is a fork of subprocess.run with fewer args.
    The goal is to create a child subprocess that is GUARANTEED to exit when the parent exits
    This is accomplished by:
    1. Making sure the child is the head of a new process group
    2. Create a third "Killer" process that is responsible for killing the child when the parent dies
    3. Killer process checks every second if the parent is dead.
    4. Killing the entire process group when we want to kill the child

    Arguments are the same as subprocess.run
    """
    # Start new session ensures that this subprocess starts as a new process group
    with subprocess.Popen(start_new_session=True, *popenargs, **kwargs) as process:
        parent_pid = os.getpid()
        child_pid = process.pid
        child_pgid = os.getpgid(child_pid)

        # Open a new subprocess to kill the child process when the parent process dies
        # kill -s 0 parent_pid will succeed if the parent is alive.
        # If it fails, SIGKILL the child process group and exit
        subprocess.Popen(
            f"while kill -s 0 {parent_pid}; do sleep 1; done; kill -9 -{child_pgid}",
            shell=True,
            # Suppress output
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except:  # noqa      (this is taken from subprocess.run directly)
            # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            # We don't call process.wait() as .__exit__ does that for us.
            raise

        retcode = process.poll()
        if check and retcode:
            raise subprocess.CalledProcessError(
                retcode, process.args, output=stdout, stderr=stderr
            )
    return subprocess.CompletedProcess(process.args, retcode or 0, stdout, stderr)

def run_background_job(command: str, env_mixin: dict) -> None:
    # Update the context with the runtime env uris
    env_vars = {
        "PYTHONUNBUFFERED": "1",  # Make sure python subprocess streams logs https://docs.python.org/3/using/cmdline.html#cmdoption-u
    }
    import os
    env = {**os.environ, **env_vars, **env_mixin}

    try:
        # TODO(mattweber): Once the publicly named run_kill_child is
        # available on product nodes, remove the underscore on this function.
        _run_kill_child(command, shell=True, check=True, env=env)  # noqa
    finally:
        # allow time for any logs to propogate before the task exits
        time.sleep(1)

if __name__ == "__main__":
    app()
