#!/usr/bin/env python

import ray
import time
import os

runtime_env = {"pip": ["jax[cpu]",]}
ray.init(runtime_env=runtime_env)


@ray.remote
class JaxWorker:
    def __init__(self):
        pass

    def get_hostname(self):
        import socket
        return socket.gethostname()

    def run_proc(self, command: str):
        print('run proc')
        run_background_job(command)


def install_jax_on_all_worker_nodes():
    create_actors_on_each_node()

def create_actors_on_each_node():
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
    hostnames = ray.get([w.get_hostname.remote() for w in workers])
    assert len(set(hostnames)) == len(hostnames), f"Hostnames not unique {hostnames}"

    return workers


def run_spmd_on_cluster():
    workers = create_actors_on_each_node()
    command = 'python3 -c "print(555)"'
    ray.get([w.run_proc.remote(command) for w in workers])

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

def run_background_job(command: str) -> None:
    # Update the context with the runtime env uris
    env_vars = {
        "PYTHONUNBUFFERED": "1",  # Make sure python subprocess streams logs https://docs.python.org/3/using/cmdline.html#cmdoption-u
    }
    import os
    env = {**os.environ, **env_vars}

    try:
        # TODO(mattweber): Once the publicly named run_kill_child is
        # available on product nodes, remove the underscore on this function.
        _run_kill_child(command, shell=True, check=True, env=env)  # noqa
    finally:
        # allow time for any logs to propogate before the task exits
        time.sleep(1)

install_jax_on_all_worker_nodes()
run_spmd_on_cluster()
