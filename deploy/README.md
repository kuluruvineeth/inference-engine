# Deploying

Two shapes, same image.

## One replica, one GPU

    docker build -t inference-engine .
    docker run --rm -p 8000:8000 inference-engine

The image runs as an unprivileged user, and the healthcheck reports
`starting` until weights are loaded, the KV cache is allocated and the
warmup pass has run. A replica that reports `starting` must not receive
traffic — cold start is tens of seconds and routing to it burns TTFT.

## Several replicas behind a cache-aware router

`ie.serve.router.CacheAwareRouter` sends a request to the replica that
already holds its prefix. Round-robin throws those hits away; measured on a
four-tenant workload the difference was 73% versus 93% prefix hit rate.

Set `load_penalty` above zero. At zero the router is maximally sticky and a
single hot tenant pins every request to one replica.

## Sizing

`ie.serve.autoscale.CapacityPlan` turns a memory budget into a concurrency
number. Feed it `ModelConfig.kv_bytes_per_token` and the block size; it
answers how many requests fit at a given context length.

`Autoscaler` reads in-flight requests and returns a replica count. Set
`scale_to_zero_after` only where a cold start is acceptable — on this
hardware that is tens of seconds before the first token.

## Placement

Prefill is compute bound, decode is memory bound (measured: step time moved
-4% while throughput rose 133x across batch 1 to 128). Colocating them means
a long prompt stalls every stream on the box. `ie.serve.disaggregated` splits
them; the cost is transferring the KV cache, and host transfer runs about
19x slower than device-to-device, so keep the two roles on the same fabric.
