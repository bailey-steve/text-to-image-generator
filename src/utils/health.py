"""Health check and monitoring utilities for production deployment."""

import logging
import time
import psutil
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class HealthChecker:
    """Health checker for monitoring application status.

    Performs various health checks including:
    - System resources (CPU, memory, disk)
    - Backend availability
    - Plugin status
    - Application uptime

    Example:
        checker = HealthChecker()
        result = checker.check_health()
        if result.status == HealthStatus.HEALTHY:
            print("Application is healthy")
    """

    def __init__(self):
        """Initialize the health checker."""
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_check_time = None

        logger.info("HealthChecker initialized")

    def check_health(self, include_details: bool = True) -> HealthCheckResult:
        """Perform comprehensive health check.

        Args:
            include_details: Whether to include detailed metrics

        Returns:
            HealthCheckResult with status and details
        """
        self.last_check_time = datetime.now()

        try:
            # Check system resources
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Determine overall health status
            status = HealthStatus.HEALTHY
            issues = []

            # Check CPU usage (warning at 80%, critical at 95%)
            if cpu_usage > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Critical CPU usage: {cpu_usage:.1f}%")
            elif cpu_usage > 80:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")

            # Check memory usage (warning at 85%, critical at 95%)
            if memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 85:
                if status != HealthStatus.UNHEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {memory.percent:.1f}%")

            # Check disk usage (warning at 85%, critical at 95%)
            if disk.percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Critical disk usage: {disk.percent:.1f}%")
            elif disk.percent > 85:
                if status != HealthStatus.UNHEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"High disk usage: {disk.percent:.1f}%")

            # Calculate uptime
            uptime_seconds = time.time() - self.start_time

            # Build result message
            if status == HealthStatus.HEALTHY:
                message = "All systems operational"
            elif status == HealthStatus.DEGRADED:
                message = f"System degraded: {', '.join(issues)}"
            else:
                message = f"System unhealthy: {', '.join(issues)}"

            # Build details dictionary
            details = {}
            if include_details:
                details = {
                    "uptime_seconds": uptime_seconds,
                    "uptime_human": self._format_uptime(uptime_seconds),
                    "cpu_usage_percent": round(cpu_usage, 2),
                    "memory_usage_percent": round(memory.percent, 2),
                    "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                    "disk_usage_percent": round(disk.percent, 2),
                    "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": round(self.error_count / max(self.request_count, 1), 4)
                }

            return HealthCheckResult(
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                details={"error": str(e)}
            )

    def check_backend(self, backend) -> HealthCheckResult:
        """Check health of a specific backend.

        Args:
            backend: Backend instance to check

        Returns:
            HealthCheckResult for the backend
        """
        try:
            is_healthy = backend.health_check()

            if is_healthy:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message=f"Backend {backend.name} is healthy",
                    details={"backend": backend.name}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Backend {backend.name} health check failed",
                    details={"backend": backend.name}
                )

        except Exception as e:
            logger.error(f"Backend health check failed for {backend.name}: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Backend {backend.name} error: {str(e)}",
                details={"backend": backend.name, "error": str(e)}
            )

    def record_request(self, success: bool = True) -> None:
        """Record a request for metrics tracking.

        Args:
            success: Whether the request was successful
        """
        self.request_count += 1
        if not success:
            self.error_count += 1

    def get_metrics(self) -> Dict:
        """Get application metrics.

        Returns:
            Dictionary with various metrics
        """
        uptime_seconds = time.time() - self.start_time

        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "uptime_seconds": uptime_seconds,
                "uptime_human": self._format_uptime(uptime_seconds),
                "cpu_usage_percent": round(cpu_usage, 2),
                "memory_usage_percent": round(memory.percent, 2),
                "memory_total_mb": round(memory.total / (1024 * 1024), 2),
                "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                "disk_usage_percent": round(disk.percent, 2),
                "disk_total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                "requests_total": self.request_count,
                "requests_failed": self.error_count,
                "error_rate": round(self.error_count / max(self.request_count, 1), 4),
                "last_check": self.last_check_time.isoformat() if self.last_check_time else None
            }

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "error": str(e),
                "uptime_seconds": uptime_seconds,
                "requests_total": self.request_count,
                "requests_failed": self.error_count
            }

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable form.

        Args:
            seconds: Uptime in seconds

        Returns:
            Formatted uptime string
        """
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def __repr__(self) -> str:
        """String representation."""
        uptime = self._format_uptime(time.time() - self.start_time)
        return f"HealthChecker(uptime={uptime}, requests={self.request_count})"


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance.

    Returns:
        Global HealthChecker instance
    """
    global _global_health_checker

    if _global_health_checker is None:
        _global_health_checker = HealthChecker()

    return _global_health_checker


def reset_health_checker() -> None:
    """Reset the global health checker instance (useful for testing)."""
    global _global_health_checker
    _global_health_checker = None
