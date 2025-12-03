"""Rate limiting utilities for API protection."""

import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RateLimitEntry:
    """Entry tracking rate limit for a client."""
    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)


class RateLimiter:
    """In-memory rate limiter for API protection.

    Implements a sliding window rate limiting algorithm.
    Each client (identified by IP or session) is limited to a certain
    number of requests within a time window.

    Attributes:
        max_requests: Maximum requests allowed per window
        window_seconds: Time window in seconds
        cleanup_interval: Seconds between cleanup of old entries

    Example:
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        # Check if request is allowed
        allowed, retry_after = limiter.is_allowed("192.168.1.1")
        if not allowed:
            print(f"Rate limited. Retry after {retry_after} seconds")
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        cleanup_interval: int = 300
    ):
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum requests per window (default: 100)
            window_seconds: Time window in seconds (default: 60)
            cleanup_interval: Seconds between cleanup (default: 300)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval

        self._clients: Dict[str, RateLimitEntry] = {}
        self._lock = Lock()
        self._last_cleanup = time.time()

        logger.info(
            f"RateLimiter initialized: {max_requests} requests per {window_seconds}s"
        )

    def is_allowed(self, client_id: str) -> Tuple[bool, Optional[int]]:
        """Check if a request from the client is allowed.

        Args:
            client_id: Unique identifier for the client (e.g., IP address)

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
            - is_allowed: True if request is allowed, False if rate limited
            - retry_after_seconds: If rate limited, seconds until next allowed request
        """
        with self._lock:
            current_time = time.time()

            # Periodic cleanup of old entries
            if current_time - self._last_cleanup > self.cleanup_interval:
                self._cleanup_old_entries(current_time)

            # Get or create client entry
            if client_id not in self._clients:
                self._clients[client_id] = RateLimitEntry(
                    request_count=1,
                    window_start=current_time,
                    last_request=current_time
                )
                logger.debug(f"New client: {client_id}")
                return True, None

            entry = self._clients[client_id]

            # Check if we need to reset the window
            time_since_window_start = current_time - entry.window_start
            if time_since_window_start >= self.window_seconds:
                # Reset window
                entry.window_start = current_time
                entry.request_count = 1
                entry.last_request = current_time
                logger.debug(f"Window reset for client: {client_id}")
                return True, None

            # Check if limit exceeded
            if entry.request_count >= self.max_requests:
                # Calculate retry-after time
                time_until_reset = self.window_seconds - time_since_window_start
                retry_after = int(time_until_reset) + 1

                logger.warning(
                    f"Rate limit exceeded for {client_id}: "
                    f"{entry.request_count}/{self.max_requests} requests. "
                    f"Retry after {retry_after}s"
                )
                return False, retry_after

            # Allow request and increment counter
            entry.request_count += 1
            entry.last_request = current_time

            logger.debug(
                f"Request allowed for {client_id}: "
                f"{entry.request_count}/{self.max_requests} in window"
            )
            return True, None

    def reset_client(self, client_id: str) -> None:
        """Reset rate limit for a specific client.

        Args:
            client_id: Client identifier to reset
        """
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"Rate limit reset for client: {client_id}")

    def get_client_status(self, client_id: str) -> Dict:
        """Get rate limit status for a client.

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with status information
        """
        with self._lock:
            if client_id not in self._clients:
                return {
                    "client_id": client_id,
                    "requests_made": 0,
                    "requests_remaining": self.max_requests,
                    "window_seconds": self.window_seconds,
                    "reset_time": None
                }

            entry = self._clients[client_id]
            current_time = time.time()
            time_since_start = current_time - entry.window_start

            if time_since_start >= self.window_seconds:
                # Window has expired but not yet reset
                requests_made = 0
                requests_remaining = self.max_requests
            else:
                requests_made = entry.request_count
                requests_remaining = max(0, self.max_requests - entry.request_count)

            reset_time = datetime.fromtimestamp(
                entry.window_start + self.window_seconds
            )

            return {
                "client_id": client_id,
                "requests_made": requests_made,
                "requests_remaining": requests_remaining,
                "window_seconds": self.window_seconds,
                "reset_time": reset_time.isoformat(),
                "last_request": datetime.fromtimestamp(entry.last_request).isoformat()
            }

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove entries for clients that haven't made requests recently.

        Args:
            current_time: Current timestamp
        """
        # Remove entries older than 2x the window size
        cutoff_time = current_time - (self.window_seconds * 2)

        clients_to_remove = [
            client_id for client_id, entry in self._clients.items()
            if entry.last_request < cutoff_time
        ]

        for client_id in clients_to_remove:
            del self._clients[client_id]

        if clients_to_remove:
            logger.info(f"Cleaned up {len(clients_to_remove)} old rate limit entries")

        self._last_cleanup = current_time

    def get_stats(self) -> Dict:
        """Get overall rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_clients = len(self._clients)
            active_clients = sum(
                1 for entry in self._clients.values()
                if time.time() - entry.last_request < self.window_seconds
            )

            return {
                "max_requests_per_window": self.max_requests,
                "window_seconds": self.window_seconds,
                "total_tracked_clients": total_clients,
                "active_clients": active_clients,
                "last_cleanup": datetime.fromtimestamp(self._last_cleanup).isoformat()
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RateLimiter(max_requests={self.max_requests}, "
            f"window_seconds={self.window_seconds}, "
            f"clients={len(self._clients)})"
        )


# Global rate limiter instance (can be configured per deployment)
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60
) -> RateLimiter:
    """Get or create the global rate limiter instance.

    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds

    Returns:
        Global RateLimiter instance
    """
    global _global_limiter

    if _global_limiter is None:
        _global_limiter = RateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds
        )

    return _global_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter instance (useful for testing)."""
    global _global_limiter
    _global_limiter = None
