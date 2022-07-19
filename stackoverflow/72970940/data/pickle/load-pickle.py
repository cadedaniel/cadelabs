#!/usr/bin/env python3

import pickle
import psutil

def print_memory_diff(label, start, end, only_show_free = False):
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

#start = psutil.virtual_memory()
#with open('no_ntlk_obj.pickle', 'rb') as f:
#    obj = pickle.load(f)
#    print(obj.func())
#end = psutil.virtual_memory()
#print_memory_diff('No nltk', start, end)

start = psutil.virtual_memory()
with open('yes_ntlk_obj.pickle', 'rb') as f:
    obj = pickle.load(f)
    print(obj.func())
end = psutil.virtual_memory()
print_memory_diff('Yes nltk', start, end)
