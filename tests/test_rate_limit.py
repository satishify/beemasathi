from api.rate_limit import TokenBucketLimiter


def test_bucket_blocks_and_refills():
    now = [0.0]
    limiter = TokenBucketLimiter(rate_per_minute=60, burst=2, clock=lambda: now[0])
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is True
    allowed, retry_after = limiter.check("a")
    assert allowed is False
    assert retry_after >= 1

    now[0] += 60
    assert limiter.check("a")[0] is True


def test_keys_are_independent():
    limiter = TokenBucketLimiter(rate_per_minute=60, burst=1, clock=lambda: 0.0)
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False
