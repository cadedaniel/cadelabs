#!/usr/bin/env python

import time
import ray

ray.init()

@ray.remote
class Tester:
    def __init__(self, param):
        self.param = param

    def run(self):
        return self.param



params = [0, 1, 2]
# I use list comprehensions instead of for loops for terseness.
testers = [Tester.remote(p) for p in params]
not_done_ids = [tester.run.remote() for tester in testers]

# len() is not required to check that the list is empty.
while not_done_ids:
    
    # Replace not_done_ids with the list of object references that aren't
    # ready. Store the list of object references that are ready in done_ids.
    # timeout=1 means sleep at most 1 second, do not sleep if there are
    # new object references that are ready.
    done_ids, not_done_ids = ray.wait(not_done_ids, timeout=1)
    
    # ray.get can take an iterable of object references.
    done_return_values = ray.get(done_ids)

    # Process each result.
    for result in done_return_values:
        print(f'result: {result}')
