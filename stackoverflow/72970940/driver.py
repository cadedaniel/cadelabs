#!/usr/bin/env python3

from dataclasses import dataclass
import subprocess

@dataclass(eq=True, frozen=True)
class TestCase:
    with_payload: bool
    with_config_obj: bool
    num_tasks: int

    def __iter__(self):
        return iter((self.with_payload, self.with_config_obj, self.num_tasks))

test_cases = [
    TestCase(with_payload=True, with_config_obj=True, num_tasks=1),
    TestCase(with_payload=True, with_config_obj=True, num_tasks=8),
    TestCase(with_payload=True, with_config_obj=True, num_tasks=16),
    TestCase(with_payload=False, with_config_obj=True, num_tasks=1),
    TestCase(with_payload=False, with_config_obj=True, num_tasks=8),
    TestCase(with_payload=False, with_config_obj=True, num_tasks=16),
    TestCase(with_payload=True, with_config_obj=False, num_tasks=1),
    TestCase(with_payload=True, with_config_obj=False, num_tasks=8),
    TestCase(with_payload=True, with_config_obj=False, num_tasks=16),
    TestCase(with_payload=False, with_config_obj=False, num_tasks=1),
    TestCase(with_payload=False, with_config_obj=False, num_tasks=8),
    TestCase(with_payload=False, with_config_obj=False, num_tasks=16),
]

outputs = {}

for with_payload, with_config_obj, num_tasks in test_cases:
    
    args = [
        './script.py',
        '--pass-payload' if with_payload else '--no-pass-payload',
        '--with-config-obj' if with_config_obj else '--without-config-obj',
        '--num-tasks', f'{num_tasks}'
    ]
    completed_proc = subprocess.run(args, capture_output=True)
    print(completed_proc.stdout.decode('utf-8'))

    used_line_filter = filter(lambda line: line.startswith(b'used'), [s.strip() for s in completed_proc.stdout.split(b'\n')])
    used_line = next(used_line_filter)
    assert next(used_line_filter, None) is None

    _, used_mb, _ = used_line.split(b' ')
    used_mb = float(used_mb)
    
    outputs[TestCase(with_payload=with_payload, with_config_obj=with_config_obj, num_tasks=num_tasks)] = used_mb

from prettytable import PrettyTable
t = PrettyTable(['with_payload', 'with_config_obj', 'num_tasks', 'used_mb', 'used_mb_per_task'])
for (with_payload, with_config_obj, num_tasks), used_mb in outputs.items():
    used_mb_per_task = round(used_mb / num_tasks, 2)
    used_mb = round(used_mb, 2)
    t.add_row([with_payload, with_config_obj, num_tasks, used_mb, used_mb_per_task])
print(t)
