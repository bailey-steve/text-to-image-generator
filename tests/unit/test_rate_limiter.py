"""Unit tests for rate limiter."""

import pytest
import time
from src.utils.rate_limiter import RateLimiter, RateLimitEntry, get_rate_limiter, reset_rate_limiter


class TestRateLimitEntry:
    """Tests for RateLimitEntry dataclass."""

    def test_create_entry(self):
        """Test creating a rate limit entry."""
        entry = RateLimitEntry()

        assert entry.request_count == 0
        assert isinstance(entry.window_start, float)
        assert isinstance(entry.last_request, float)

    def test_entry_with_values(self):
        """Test creating entry with custom values."""
        now = time.time()
        entry = RateLimitEntry(
            request_count=5,
            window_start=now,
            last_request=now
        )

        assert entry.request_count == 5
        assert entry.window_start == now
        assert entry.last_request == now


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_rate_limiter()

    def teardown_method(self):
        """Clean up after tests."""
        reset_rate_limiter()

    def test_initialization(self):
        """Test limiter initialization."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60
        assert limiter.cleanup_interval == 300

    def test_first_request_allowed(self):
        """Test that first request from client is allowed."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        allowed, retry_after = limiter.is_allowed("client1")

        assert allowed is True
        assert retry_after is None

    def test_multiple_requests_within_limit(self):
        """Test multiple requests within rate limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Make 5 requests (at limit)
        for i in range(5):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after is None

    def test_request_exceeds_limit(self):
        """Test request that exceeds rate limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # Make 3 requests (at limit)
        for i in range(3):
            limiter.is_allowed("client1")

        # 4th request should be denied
        allowed, retry_after = limiter.is_allowed("client1")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_different_clients_independent(self):
        """Test that different clients have independent limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Client 1 makes 2 requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Client 2 should still be allowed
        allowed, retry_after = limiter.is_allowed("client2")

        assert allowed is True
        assert retry_after is None

    def test_window_reset(self):
        """Test that window resets after time expires."""
        # Use very short window for testing
        limiter = RateLimiter(max_requests=2, window_seconds=1)

        # Make 2 requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Should be at limit
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is True
        assert retry_after is None

    def test_reset_client(self):
        """Test resetting rate limit for specific client."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Make 2 requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Should be at limit
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Reset client
        limiter.reset_client("client1")

        # Should be allowed again
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is True
        assert retry_after is None

    def test_reset_nonexistent_client(self):
        """Test resetting client that doesn't exist."""
        limiter = RateLimiter()

        # Should not raise error
        limiter.reset_client("nonexistent")

    def test_get_client_status_new_client(self):
        """Test getting status for new client."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        status = limiter.get_client_status("client1")

        assert status["client_id"] == "client1"
        assert status["requests_made"] == 0
        assert status["requests_remaining"] == 10
        assert status["window_seconds"] == 60
        assert status["reset_time"] is None

    def test_get_client_status_existing_client(self):
        """Test getting status for existing client."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        # Make some requests
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        status = limiter.get_client_status("client1")

        assert status["client_id"] == "client1"
        assert status["requests_made"] == 3
        assert status["requests_remaining"] == 7
        assert status["reset_time"] is not None
        assert status["last_request"] is not None

    def test_cleanup_old_entries(self):
        """Test cleanup of old client entries."""
        # Use short window for testing
        limiter = RateLimiter(max_requests=10, window_seconds=1)

        # Make request from client
        limiter.is_allowed("client1")

        assert "client1" in limiter._clients

        # Wait for entry to become old
        time.sleep(2.5)

        # Trigger cleanup by making request from different client
        limiter._cleanup_old_entries(time.time())

        # Old client should be removed
        assert "client1" not in limiter._clients

    def test_get_stats(self):
        """Test getting overall statistics."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        # Make requests from multiple clients
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")
        limiter.is_allowed("client3")

        stats = limiter.get_stats()

        assert stats["max_requests_per_window"] == 10
        assert stats["window_seconds"] == 60
        assert stats["total_tracked_clients"] == 3
        assert stats["active_clients"] == 3
        assert "last_cleanup" in stats

    def test_repr(self):
        """Test string representation."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.is_allowed("client1")

        repr_str = repr(limiter)

        assert "RateLimiter" in repr_str
        assert "10" in repr_str
        assert "60" in repr_str

    def test_global_rate_limiter(self):
        """Test global rate limiter singleton."""
        limiter1 = get_rate_limiter(max_requests=10, window_seconds=60)
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2

    def test_reset_global_rate_limiter(self):
        """Test resetting global rate limiter."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is not limiter2

    def test_concurrent_requests_same_client(self):
        """Test handling concurrent requests from same client."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Simulate concurrent requests
        results = []
        for i in range(7):
            allowed, retry_after = limiter.is_allowed("client1")
            results.append(allowed)

        # First 5 should be allowed, rest denied
        assert sum(results) == 5
        assert results[:5] == [True] * 5
        assert results[5:] == [False] * 2
