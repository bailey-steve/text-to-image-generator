"""Unit tests for health checker."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils.health import (
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
    get_health_checker,
    reset_health_checker
)


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="All systems operational"
        )

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.details == {}
        assert result.timestamp is not None

    def test_result_with_details(self):
        """Test result with details."""
        details = {"cpu": 50.5, "memory": 70.2}
        result = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message="High resource usage",
            details=details
        )

        assert result.status == HealthStatus.DEGRADED
        assert result.details == details

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="OK",
            details={"uptime": 3600}
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "OK"
        assert result_dict["details"]["uptime"] == 3600
        assert "timestamp" in result_dict


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_health_checker()

    def teardown_method(self):
        """Clean up after tests."""
        reset_health_checker()

    def test_initialization(self):
        """Test health checker initialization."""
        checker = HealthChecker()

        assert checker.request_count == 0
        assert checker.error_count == 0
        assert checker.start_time > 0
        assert checker.last_check_time is None

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_healthy(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test health check with healthy system."""
        # Mock low resource usage
        mock_cpu.return_value = 20.0
        mock_memory.return_value = Mock(percent=30.0, available=8000000000)
        mock_disk.return_value = Mock(percent=40.0, free=500000000000)

        checker = HealthChecker()
        result = checker.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert "operational" in result.message.lower()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_degraded_cpu(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test health check with high CPU usage."""
        mock_cpu.return_value = 85.0  # High but not critical
        mock_memory.return_value = Mock(percent=30.0, available=8000000000)
        mock_disk.return_value = Mock(percent=40.0, free=500000000000)

        checker = HealthChecker()
        result = checker.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert "cpu" in result.message.lower()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_unhealthy_memory(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test health check with critical memory usage."""
        mock_cpu.return_value = 20.0
        mock_memory.return_value = Mock(percent=96.0, available=1000000)  # Critical
        mock_disk.return_value = Mock(percent=40.0, free=500000000000)

        checker = HealthChecker()
        result = checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "memory" in result.message.lower()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_unhealthy_disk(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test health check with critical disk usage."""
        mock_cpu.return_value = 20.0
        mock_memory.return_value = Mock(percent=30.0, available=8000000000)
        mock_disk.return_value = Mock(percent=97.0, free=1000000)  # Critical

        checker = HealthChecker()
        result = checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "disk" in result.message.lower()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_with_details(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test health check includes detailed metrics."""
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=45.3, available=4000000000)
        mock_disk.return_value = Mock(percent=60.2, free=200000000000)

        checker = HealthChecker()
        result = checker.check_health(include_details=True)

        assert "cpu_usage_percent" in result.details
        assert "memory_usage_percent" in result.details
        assert "disk_usage_percent" in result.details
        assert "uptime_seconds" in result.details
        assert "request_count" in result.details
        assert "error_count" in result.details

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_without_details(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test health check without detailed metrics."""
        mock_cpu.return_value = 20.0
        mock_memory.return_value = Mock(percent=30.0, available=8000000000)
        mock_disk.return_value = Mock(percent=40.0, free=500000000000)

        checker = HealthChecker()
        result = checker.check_health(include_details=False)

        assert result.details == {}

    @patch('psutil.cpu_percent', side_effect=Exception("System error"))
    def test_check_health_error(self, mock_cpu):
        """Test health check handles errors gracefully."""
        checker = HealthChecker()
        result = checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert "error" in result.message.lower()

    def test_check_backend_healthy(self):
        """Test checking a healthy backend."""
        checker = HealthChecker()

        # Mock backend
        mock_backend = Mock()
        mock_backend.name = "TestBackend"
        mock_backend.health_check.return_value = True

        result = checker.check_backend(mock_backend)

        assert result.status == HealthStatus.HEALTHY
        assert "TestBackend" in result.message
        assert result.details["backend"] == "TestBackend"

    def test_check_backend_unhealthy(self):
        """Test checking an unhealthy backend."""
        checker = HealthChecker()

        # Mock backend
        mock_backend = Mock()
        mock_backend.name = "TestBackend"
        mock_backend.health_check.return_value = False

        result = checker.check_backend(mock_backend)

        assert result.status == HealthStatus.UNHEALTHY
        assert "TestBackend" in result.message

    def test_check_backend_error(self):
        """Test checking a backend that raises an error."""
        checker = HealthChecker()

        # Mock backend that raises error
        mock_backend = Mock()
        mock_backend.name = "TestBackend"
        mock_backend.health_check.side_effect = Exception("Connection failed")

        result = checker.check_backend(mock_backend)

        assert result.status == HealthStatus.UNHEALTHY
        assert "error" in result.message.lower()
        assert "error" in result.details

    def test_record_request_success(self):
        """Test recording successful request."""
        checker = HealthChecker()

        checker.record_request(success=True)

        assert checker.request_count == 1
        assert checker.error_count == 0

    def test_record_request_failure(self):
        """Test recording failed request."""
        checker = HealthChecker()

        checker.record_request(success=False)

        assert checker.request_count == 1
        assert checker.error_count == 1

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        checker = HealthChecker()

        checker.record_request(success=True)
        checker.record_request(success=True)
        checker.record_request(success=False)
        checker.record_request(success=True)

        assert checker.request_count == 4
        assert checker.error_count == 1

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_metrics(
        self,
        mock_disk,
        mock_memory,
        mock_cpu
    ):
        """Test getting application metrics."""
        mock_cpu.return_value = 30.5
        mock_memory.return_value = Mock(
            percent=50.2,
            total=16000000000,
            available=8000000000
        )
        mock_disk.return_value = Mock(
            percent=65.3,
            total=1000000000000,
            free=350000000000
        )

        checker = HealthChecker()
        checker.request_count = 100
        checker.error_count = 5

        metrics = checker.get_metrics()

        assert metrics["uptime_seconds"] > 0
        assert metrics["cpu_usage_percent"] == 30.5
        assert metrics["memory_usage_percent"] == 50.2
        assert metrics["disk_usage_percent"] == 65.3
        assert metrics["requests_total"] == 100
        assert metrics["requests_failed"] == 5
        assert metrics["error_rate"] == 0.05

    @patch('psutil.cpu_percent', side_effect=Exception("Error"))
    def test_get_metrics_error(self, mock_cpu):
        """Test get_metrics handles errors gracefully."""
        checker = HealthChecker()

        metrics = checker.get_metrics()

        assert "error" in metrics
        assert "uptime_seconds" in metrics

    def test_format_uptime(self):
        """Test uptime formatting."""
        checker = HealthChecker()

        # Test various durations
        assert "30s" in checker._format_uptime(30)
        assert "1m" in checker._format_uptime(60)
        assert "1m 30s" in checker._format_uptime(90)
        assert "1h" in checker._format_uptime(3600)
        assert "1h 30m" in checker._format_uptime(5400)
        assert "1d" in checker._format_uptime(86400)
        assert "2d 3h" in checker._format_uptime(183600)

    def test_repr(self):
        """Test string representation."""
        checker = HealthChecker()
        checker.request_count = 150

        repr_str = repr(checker)

        assert "HealthChecker" in repr_str
        assert "uptime=" in repr_str
        assert "requests=150" in repr_str

    def test_global_health_checker(self):
        """Test global health checker singleton."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

    def test_reset_global_health_checker(self):
        """Test resetting global health checker."""
        checker1 = get_health_checker()
        reset_health_checker()
        checker2 = get_health_checker()

        assert checker1 is not checker2
