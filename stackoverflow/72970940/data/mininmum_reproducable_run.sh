#!/usr/bin/env bash

# Ray 1.13, Deep Learning AMI (Ubuntu 18.04) Version 63.0, c5.4xlarge (16cpu)

./script.py --no-pass-payload --without-config-obj --num-tasks 16 --payload-size-mb 10
./script.py --no-pass-payload --with-config-obj --num-tasks 1 --payload-size-mb 10
./script.py --no-pass-payload --with-config-obj --num-tasks 8 --payload-size-mb 10
./script.py --no-pass-payload --with-config-obj --num-tasks 16 --payload-size-mb 10
