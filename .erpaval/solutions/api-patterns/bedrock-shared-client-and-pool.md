---
name: Bedrock shared client + max_pool_connections is load-bearing at concurrency
description: A single shared boto3 bedrock-runtime client with tuned Config unblocks any concurrency > 10; botocore's pool default starves workers silently.
type: reference
---

One `boto3.client('bedrock-runtime', ...)` is thread-safe (boto3 docs,
2026-05) and is intended to be shared across workers. Creating a client
per request wastes the TCP pool and the configured adaptive-retry token
bucket, and adds ~50 ms setup per call.

The silent bottleneck is `max_pool_connections`: botocore default is
**10**. At `concurrency=16`, six workers serialize on urllib3's pool
lock with no log line explaining the wait. AWS's Bedrock scale guide
recommends 50 for high-throughput workloads; safe formula is
`max(32, 2 × max(embed_conc, llm_conc))`.

Full recommended config for Bedrock `invoke_model` at concurrency:

```python
import threading
import boto3
from botocore.config import Config

_LOCK = threading.Lock()
_CACHE: dict[tuple[str, int], object] = {}

def build_client(region: str, max_conc: int) -> object:
    pool = max(32, max_conc * 2)
    key = (region, pool)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = boto3.client(
                "bedrock-runtime",
                config=Config(
                    region_name=region,
                    max_pool_connections=pool,
                    connect_timeout=10,
                    read_timeout=600,   # Sonnet adaptive thinking can run past 60s
                    retries={"max_attempts": 0, "mode": "adaptive"},
                ),
            )
        return _CACHE[key]
```

Why `max_attempts=0` with `mode='adaptive'`:
  - Adaptive's client-side token bucket still runs, absorbing 429 bursts at the SDK layer.
  - `max_attempts=0` disables botocore's own retry loop so app-level retries (tenacity) see errors immediately.
  - Two-tier retry: adaptive throttle-shaping at the SDK + semantic classification at the app.

Extended retry codes for Bedrock (per AWS re:Post 2026-05):
`ThrottlingException`, `ServiceUnavailableException`, `ModelTimeoutException`,
`ModelErrorException`, `ProvisionedThroughputExceededException`,
`TooManyRequestsException`, `InternalServerException`, `InternalFailure`.

**When you hit this**: concurrent Bedrock workload with no visible throttling
but fewer actual parallel requests than `concurrency` would imply. Check
`max_pool_connections` first.
