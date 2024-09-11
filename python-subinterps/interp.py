#!/usr/bin/env python3

import _xxsubinterpreters as interpreters

import os
#import psutil
import threading
import ctypes

def set_affinity(tid, core_id):
	libc = ctypes.CDLL('libc.so.6', use_errno=True)
	mask = ctypes.c_ulong(1 << core_id)
	result = libc.sched_setaffinity(tid, ctypes.sizeof(mask), ctypes.byref(mask))
	if result != 0:
		errno = ctypes.get_errno()
		raise OSError(errno, f"Error in sched_setaffinity: {os.strerror(errno)}")

code = """
import random


N = 100
loops = 0

matrix1 = [[random.random() for _ in range(N)] for _ in range(N)]
matrix2 = [[random.random() for _ in range(N)] for _ in range(N)]

for i in range(20):
	# Initialize the result matrix with zeros
	result = [[0 for _ in range(N)] for _ in range(N)]
	
	# Multiply the matrices
	for i in range(N):
		for j in range(N):
			for k in range(N):
				result[i][j] += matrix1[i][k] * matrix2[k][j]

	loops += 1
	print(f'finished loop {loops=}')
"""
#interpreters.run_string(subinterp, code)

from threading import Thread

subinterps = [interpreters.create() for _ in range(4)]

threads = [Thread(target=interpreters.run_string, args=(subinterp, code))
	for subinterp in subinterps]


import time
start_time = time.time()
[t.start() for t in threads]

assignment = {t.native_id: t.native_id % 8 for t in threads}
print(f'{assignment=}')
[set_affinity(tid, core) for tid, core in assignment.items()]

[t.join() for t in threads]

end_time = time.time()
[interpreters.destroy(subinterp) for subinterp in subinterps]

dur_s = end_time - start_time
print(f'{dur_s=:.02f}')


