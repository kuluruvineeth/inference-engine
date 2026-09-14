import random

import pytest

from ie.serve.router import CacheAwareRouter, ReplicaIndex, RoundRobinRouter

BS = 16


def tenant(index, length=96):
    return list(range(1000 + index * 500, 1000 + index * 500 + length))


def workload(num_tenants=4, count=60, seed=0):
    generator = random.Random(seed)
    tenants = [tenant(i) for i in range(num_tenants)]
    return [tenants[generator.randrange(num_tenants)] + [generator.randrange(9000, 9999)]
            for _ in range(count)]


def test_a_replica_recognises_a_prefix_it_has_seen():
    replica = ReplicaIndex(0, BS)
    prompt = tenant(0)
    assert replica.matching_blocks(prompt) == 0
    replica.remember(prompt)
    assert replica.matching_blocks(prompt) == len(prompt) // BS


def test_a_replica_matches_only_the_shared_leading_blocks():
    replica = ReplicaIndex(0, BS)
    replica.remember(tenant(0))
    diverged = tenant(0)[:BS] + [7] * BS
    assert replica.matching_blocks(diverged) == 1


def test_round_robin_spreads_evenly():
    router = RoundRobinRouter(num_replicas=4, block_size=BS)
    for index in range(8):
        assert router.route(tenant(index)) == index % 4


def test_router_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        RoundRobinRouter(num_replicas=0, block_size=BS)


def test_cache_aware_sends_a_repeat_back_to_the_same_replica():
    router = CacheAwareRouter(num_replicas=4, block_size=BS, load_penalty=0.0)
    prompt = tenant(0)
    first = router.route(prompt)
    assert router.route(prompt + [42]) == first


def test_cache_aware_beats_round_robin_on_a_multi_tenant_workload():
    requests = workload()

    spread = RoundRobinRouter(num_replicas=4, block_size=BS)
    for request in requests:
        spread.route(request)

    aware = CacheAwareRouter(num_replicas=4, block_size=BS)
    for request in requests:
        aware.route(request)

    assert aware.stats.hit_rate > spread.stats.hit_rate
    assert aware.stats.blocks_matched > spread.stats.blocks_matched


def test_the_load_penalty_keeps_replicas_from_starving():
    requests = [tenant(0) + [i] for i in range(40)]

    sticky = CacheAwareRouter(num_replicas=4, block_size=BS, load_penalty=0.0)
    balanced = CacheAwareRouter(num_replicas=4, block_size=BS, load_penalty=1.0)
    for request in requests:
        sticky.route(request)
        balanced.route(request)

    sticky_load = [r.in_flight for r in sticky.replicas]
    balanced_load = [r.in_flight for r in balanced.replicas]

    assert max(sticky_load) == len(requests)
    assert max(balanced_load) < len(requests)


def test_releasing_a_request_frees_capacity():
    router = CacheAwareRouter(num_replicas=2, block_size=BS)
    chosen = router.route(tenant(0))
    assert router.replicas[chosen].in_flight == 1
    router.release(chosen)
    assert router.replicas[chosen].in_flight == 0
