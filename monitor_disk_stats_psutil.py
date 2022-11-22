#!/usr/bin/env python3

import psutil
import time

while True:

    io_counters = psutil.disk_io_counters()
    print('disk_io_counters[readbytes]', io_counters.read_bytes)

    io_counters = psutil.disk_io_counters(perdisk=True)
    for k, v in io_counters.items():
        print(f'disk_io_counters[{k}][readbytes]', v.read_bytes)

    print('=' * 20)
    time.sleep(5)
