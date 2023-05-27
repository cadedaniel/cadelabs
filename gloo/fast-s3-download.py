#!/usr/bin/env python3

"""
pip install pycurl
sudo apt-get install libcurl4-openssl-dev
"""

import os
import mmap

#print(mmap.PAGESIZE)

def main():
    size_bytes = 9943729448
    url = "https://large-dl-models-mirror.s3.us-west-2.amazonaws.com/models--amazon--LightGPT/main-safetensors/model-00001-of-00002.safetensors"
    dest_file = '/mnt/local_storage/out'

    target_bandwidth_mb_s = 5 * 1024
    single_stream_bandwidth_mb_s = 32

    align = False

    ranges = generate_ranges(size_bytes, target_bandwidth_mb_s, single_stream_bandwidth_mb_s, mmap.PAGESIZE if align else None)

    import time

    try:
        print(f'Removing {dest_file}')
        os.remove(dest_file)
    except FileNotFoundError:
        pass

    # Prepare file
    with open(dest_file, 'wb') as f:
        f.truncate(size_bytes)

    while True:
        with open(dest_file, 'r+b') as f:
            read_into_mmap_mp(f.fileno(), ranges, url, size_bytes, f)


    

class ShmFileWrapper:
    def __init__(self, shm):
        self.memview = shm.buf
        self.offset = 0

    def seek(self, offset):
        self.offset = offset

    def write(self, data):
        old_offset = self.offset
        new_offset = self.offset + len(data)
        self.memview[old_offset:new_offset] = data
        self.offset = new_offset
        
        return len(data)

def read_range_into_mmap_sp(args):
    try:
        return read_range_into_mmap_sp_inner(args)
    except Exception as e:
        return e

def read_range_into_mmap_sp_inner(args):
    import mmap
    import pycurl
    import certifi
    from io import BytesIO

    fileno, start_index, end_index, url = args
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=fileno, create=False)

    wrapper = ShmFileWrapper(shm)
    wrapper.seek(start_index)

    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, wrapper)
    c.setopt(c.CAINFO, certifi.where())
    c.setopt(c.ENCODING, 'gzip,deflate')
    c.setopt(c.RANGE, f'{start_index}-{end_index}')
    c.perform()
    c.close()


def read_into_mmap_mp(fileno, ranges, url, total_size, dest_file_obj):
    import multiprocessing as mp
    from multiprocessing import shared_memory
    import pycurl
    import certifi
    import time

    try:
        shm = shared_memory.SharedMemory(create=True, size=total_size)
        
        #mm = mmap.mmap(shm.fd, shm.size)
        #fileno = shm.fd

        print(f'creating {len(ranges)} processes')

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(ranges)) as p:
            print(f'pool created {p}')
            start = time.time()

            args = [(shm.name, start_index, end_index, url) for start_index, end_index in ranges]
            results = p.imap_unordered(read_range_into_mmap_sp, args, chunksize=1)

            for r in results:
                assert r == None, r
            
            dur_s = time.time() - start
            print(f'network dur_s {dur_s:.02f} speed {total_size/(dur_s*2**30):.02f} GB/s')

    finally:
        shm.close()
        shm.unlink()


def generate_ranges(size_bytes, target_bandwidth_mb_s, single_stream_bandwidth_mb_s, mmap_pagesize=None):
    num_shards = target_bandwidth_mb_s // single_stream_bandwidth_mb_s
    bytes_per_shard = size_bytes // num_shards

    if mmap_pagesize is not None:
        bytes_per_shard = mmap_pagesize * round(bytes_per_shard / mmap_pagesize)
    
    ranges = []
    
    start_index = 0
    end_index = bytes_per_shard
    done = False
    while not done:
        if end_index > size_bytes:
            end_index = size_bytes
            done = True
    
        new_range = (start_index, end_index)
        ranges.append(new_range)
    
        start_index = end_index
        end_index += bytes_per_shard
    
        if start_index > size_bytes:
            break

    return ranges


if __name__ == '__main__':
    main()
