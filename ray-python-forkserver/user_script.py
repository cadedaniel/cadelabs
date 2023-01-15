#!/usr/bin/python3
#import slow_import
import os
import time
import sys

print(f'User script loaded, pid={os.getpid()}. Sleeping forever.')
sys.stdout.flush()

while True:
    time.sleep(1)
