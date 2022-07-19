#!/usr/bin/env python3

import pickle
import psutil
import multiprocessing
import functools

def main():
    for num_tasks in [2**p for p in range(7)]:
        test_with_and_without(num_tasks)

def test_with_and_without(num_tasks):
    num_procs = num_tasks
    assert num_tasks <= num_procs

    m = multiprocessing.Manager()
    worker_input_queue = m.Queue(maxsize=num_tasks)
    worker_output_queue = m.Queue(maxsize=num_tasks)
    
    # Pickle the classes in another process, so that nltk is not imported in the driver process,
    # since forking Copy-on-Write would eliminate the additional memory overhead.
    with multiprocessing.Pool(processes=1) as pool:
        pickle_without_nltk, pickle_with_nltk = pool.apply(create_pickle_data)

    print(f'num_tasks {num_tasks}')
    for test_label, test_pickle_data in [('Pickle without nltk', pickle_without_nltk), ('Pickle with nltk', pickle_with_nltk)]:
        with multiprocessing.Pool(processes=num_procs) as pool:
            start = psutil.virtual_memory()
            pool.map_async(functools.partial(test_load_from_pickle, test_pickle_data, worker_input_queue, worker_output_queue), range(num_tasks))
        
            for _ in range(num_tasks):
                worker_output_queue.get()
        
            end = psutil.virtual_memory()
            print_memory_diff(test_label, start, end, num_tasks)
        
            for _ in range(num_tasks):
                worker_input_queue.put(None)

def create_pickle_data():
    import nltk

    class WithoutNltk:
        def func(self):
            return 'without nltk'
    
    class WithNltk:
        def func(self):
            return nltk.__version__

    # Make sure they work.
    WithoutNltk().func()
    WithNltk().func()
    
    import cloudpickle
    return cloudpickle.dumps(WithoutNltk()), cloudpickle.dumps(WithNltk())


def test_load_from_pickle(pickle_data, worker_input_queue, worker_output_queue, rank):
    obj = pickle.loads(pickle_data)
    obj.func()

    worker_output_queue.put(rank)
    worker_input_queue.get()


def print_memory_diff(label, start, end, num_tasks):
    unit_symbol = 'MB'
    bytes_per_unit = 1 << 20

    lines = [
            f'{label} memory difference',
            #f'total {(end.total - start.total) / bytes_per_unit:+} {unit_symbol}',
            #f'available {(end.available - start.available) / bytes_per_unit:+} {unit_symbol}',
            f'used {(end.used - start.used) / bytes_per_unit:+} {unit_symbol}',
            f'used_per {(end.used - start.used) / (bytes_per_unit * num_tasks):+} {unit_symbol}',
            #f'free {(end.free - start.free) / bytes_per_unit:+} {unit_symbol}',
            #f'active {(end.active - start.active) / bytes_per_unit:+} {unit_symbol}',
            #f'inactive {(end.inactive - start.inactive) / bytes_per_unit:+} {unit_symbol}',
            #f'shared {(end.shared - start.shared) / bytes_per_unit:+} {unit_symbol}',
    ]

    print('\n\t'.join(lines))

if __name__ == '__main__':
    main()
