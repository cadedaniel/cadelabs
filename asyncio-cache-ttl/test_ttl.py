#!/usr/bin/env python3

import asyncio
from cachetools import TTLCache
from asyncache import cached

# Need to see if it's in cache, if so then return
# If it's not in cache, then call function

class K8sClient:
    pass

async def get_k8s_client(config):
    await asyncio.sleep(1)
    return K8sClient()


@cached(TTLCache(maxsize=2**10, ttl=60*60))
async def get_k8s_client_cached(config):
    k8s_client = await get_k8s_client(config)
    return k8s_client

async def test():
    client_config = 'key'

    for _ in range(5):
        client = await get_k8s_client_cached(config=client_config)
        print(f'got client {client}')

loop = asyncio.get_event_loop()
loop.run_until_complete(test())


