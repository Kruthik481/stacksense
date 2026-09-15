from rate_limit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestSlidingWindowRateLimiter:
    def test_allows_up_to_limit(self):
        limiter = SlidingWindowRateLimiter(window_seconds=60, clock=FakeClock())
        assert [limiter.allow("a", limit=2) for _ in range(3)] == [True, True, False]

    def test_window_expiry_frees_capacity(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(window_seconds=60, clock=clock)
        limiter.allow("a", limit=1)
        assert not limiter.allow("a", limit=1)
        clock.now += 61
        assert limiter.allow("a", limit=1)

    def test_keys_are_isolated(self):
        limiter = SlidingWindowRateLimiter(window_seconds=60, clock=FakeClock())
        assert limiter.allow("a", limit=1)
        assert limiter.allow("b", limit=1)

    def test_reset_clears_state(self):
        limiter = SlidingWindowRateLimiter(window_seconds=60, clock=FakeClock())
        limiter.allow("a", limit=1)
        limiter.reset()
        assert limiter.allow("a", limit=1)
