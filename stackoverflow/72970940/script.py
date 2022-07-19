#!/usr/bin/env python3

import nltk
import psutil
import ray
import click
import sys

'''
tasks: 1, 8, 16
payload0_config0: +0, +0.89, +2.98
payload0_config1: +18.6 +132.1 +257.49
payload1_config0: +1.93 +14.38 +30.8
payload1_config1: +27.6 +210.1 +415.9

0.0, 0.11, 0.19
16.5, 16.5, 16.1
1.93, 1.79, 1.93
27.6, 26.26, 25.99

'''

class DummyObject:
    def do_something(self):
        print(nltk.__version__)


@ray.remote
def dummy_fun(config, payload):

    return type(config), type(payload)


def create_data(target_size_mb):
    bytes_per_mb = 1 << 20
    payload = 'a' * target_size_mb * bytes_per_mb
    return payload


def run_problem(pass_payload: bool, with_config_obj: bool, num_tasks: int, payload_size_mb: int, only_show_free: bool) -> None:
    # Init ray
    ray.init()
    
    payload = create_data(payload_size_mb)
    config = DummyObject()

    payload_id = None
    config_id = None

    if pass_payload:
        payload_id = ray.put(payload)

    if with_config_obj:
        config_id = ray.put(config)

    # Track memory in a naive way
    start_memory = psutil.virtual_memory()

    # Create jobs
    result_id = [dummy_fun.remote(config_id, payload_id) for _ in range(num_tasks)]

    # Run jobs
    result = ray.get(result_id)

    end_memory = psutil.virtual_memory()
    print_memory_diff('', start_memory, end_memory, only_show_free)

    ray.shutdown()

def print_memory_diff(label, start, end, only_show_free):
    unit_symbol = 'MB'
    bytes_per_unit = 1 << 20

    free_line = f'free {(end.free - start.free) / bytes_per_unit:+} {unit_symbol}'
    lines = [f'{label} memory difference']

    if only_show_free:
        lines += [free_line]
    else:
        lines += [f'total {(end.total - start.total) / bytes_per_unit:+} {unit_symbol}',
            f'available {(end.available - start.available) / bytes_per_unit:+} {unit_symbol}',
            f'used {(end.used - start.used) / bytes_per_unit:+} {unit_symbol}',
            free_line,
            f'active {(end.active - start.active) / bytes_per_unit:+} {unit_symbol}',
            f'inactive {(end.inactive - start.inactive) / bytes_per_unit:+} {unit_symbol}',
            f'shared {(end.shared - start.shared) / bytes_per_unit:+} {unit_symbol}',
        ]

    print('\n\t'.join(lines))


@click.command()
@click.option('--pass-payload/--no-pass-payload', default=True)
@click.option('--with-config-obj/--without-config-obj', default=True)
@click.option('--num-tasks', default=16)
@click.option('--payload-size-mb', default=10)
def cli(pass_payload: bool, with_config_obj: bool, num_tasks: int, payload_size_mb: int, only_show_free: bool = False):
    print(f'Pass payload? {pass_payload}, Pass config object? {with_config_obj}, Number of tasks {num_tasks}, Payload size (MB) {payload_size_mb}')
    run_problem(pass_payload, with_config_obj, num_tasks, payload_size_mb, only_show_free)

if __name__ == '__main__':
    sys.exit(cli())
