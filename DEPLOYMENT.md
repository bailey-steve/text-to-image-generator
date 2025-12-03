# Production Deployment Guide

This guide covers deploying the Text-to-Image Generator in production environments.

## Table of Contents

1. [Quick Start with Docker](#quick-start-with-docker)
2. [Docker Deployment](#docker-deployment)
3. [Configuration](#configuration)
4. [Monitoring & Health Checks](#monitoring--health-checks)
5. [Rate Limiting](#rate-limiting)
6. [Security Best Practices](#security-best-practices)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

## Quick Start with Docker

The fastest way to deploy in production:

```bash
# 1. Clone the repository
git clone https://github.com/bailey-steve/text-to-image-generator.git
cd text-to-image-generator

# 2. Create .env file with your API keys
cp .env.example .env
# Edit .env and add your tokens

# 3. Start with Docker Compose
docker-compose up -d

# 4. Check status
docker-compose ps
docker-compose logs -f app
```

The application will be available at `http://localhost:7860`

## Docker Deployment

### Building the Docker Image

```bash
# Build the image
docker build -t text-to-image-generator:latest .

# Run the container
docker run -d \
  --name text-to-image-app \
  -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  text-to-image-generator:latest
```

### Using Docker Compose (Recommended)

Docker Compose provides easier management and includes optional services:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop all services
docker-compose down

# Restart a specific service
docker-compose restart app

# View resource usage
docker-compose stats
```

### Docker Compose Services

The `docker-compose.yml` includes:

**Main Application** (always enabled):
- Text-to-image generation service
- Health checks
- Automatic restart
- Volume mounts for data persistence

**Optional Services** (commented out by default):
- **Redis**: For distributed rate limiting
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization dashboard

To enable optional services, uncomment them in `docker-compose.yml`

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# API Tokens
HUGGINGFACE_TOKEN=hf_your_token_here
REPLICATE_TOKEN=r8_your_token_here  # Optional

# Backend Configuration
DEFAULT_BACKEND=huggingface
FALLBACK_BACKEND=replicate
ENABLE_FALLBACK=true

# Application Settings
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT=60

# Production Settings
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100  # Requests per window
RATE_LIMIT_WINDOW=60     # Window in seconds
ENABLE_HEALTH_CHECKS=true
ENABLE_METRICS=true
PRODUCTION_MODE=true

# Local Backend (if using)
LOCAL_MODEL=stabilityai/sd-turbo
```

### Production Mode

Set `PRODUCTION_MODE=true` to enable:
- Stricter error handling
- More comprehensive logging
- Enhanced security measures
- Performance optimizations

### Gradio Configuration

Additional Gradio-specific environment variables:

```bash
# Server settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860

# Security (if needed)
GRADIO_AUTH=username:password  # Basic auth
GRADIO_SHARE=false             # Don't create public link
```

## Monitoring & Health Checks

### Health Check Endpoint

The application exposes a `/health` endpoint for monitoring:

```bash
curl http://localhost:7860/health
```

Response format:
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "details": {
    "uptime_seconds": 3600,
    "uptime_human": "1h 0m 0s",
    "cpu_usage_percent": 15.2,
    "memory_usage_percent": 45.3,
    "disk_usage_percent": 60.1,
    "request_count": 150,
    "error_count": 2,
    "error_rate": 0.0133
  },
  "timestamp": "2025-12-03T18:30:00"
}
```

### Health Status Levels

- **healthy**: All systems operational
- **degraded**: Some issues detected (high resource usage)
- **unhealthy**: Critical issues (system overloaded, disk full)

### Docker Health Checks

Docker automatically monitors application health:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' text-to-image-app

# View health check logs
docker inspect text-to-image-app | jq '.[].State.Health'
```

### Load Balancer Integration

Configure your load balancer to use `/health` for health checks:

**nginx** example:
```nginx
upstream app_servers {
    server 127.0.0.1:7860 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:7861 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;

    location / {
        proxy_pass http://app_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /health {
        proxy_pass http://app_servers/health;
        access_log off;
    }
}
```

## Rate Limiting

### Configuration

Rate limiting protects against abuse:

```bash
# In .env
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100  # Max requests per client
RATE_LIMIT_WINDOW=60     # Time window in seconds
```

### How It Works

- Each client (identified by IP) is limited to `RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW`
- Uses sliding window algorithm
- In-memory storage (single instance)
- For distributed deployment, use Redis (see Advanced section)

### Rate Limit Response

When rate limited, clients receive:
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 45,
  "message": "Too many requests. Try again in 45 seconds"
}
```

### Advanced: Redis-Based Rate Limiting

For multi-instance deployments:

1. Uncomment Redis service in `docker-compose.yml`
2. Install Redis client: `pip install redis`
3. Configure application to use Redis (future enhancement)

## Security Best Practices

### 1. Environment Variables

**Never commit `.env` files to version control:**

```bash
# Add to .gitignore (already included)
.env
.env.local
.env.*.local
```

### 2. API Key Management

- Use environment variables for API keys
- Rotate keys regularly
- Use separate keys for dev/staging/production
- Monitor key usage for anomalies

### 3. Network Security

**Use HTTPS in production:**

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:7860;
    }
}
```

### 4. Container Security

```dockerfile
# Run as non-root user (already configured)
USER appuser

# Scan images for vulnerabilities
docker scan text-to-image-generator:latest
```

### 5. Authentication

Enable Gradio authentication:

```bash
# In .env
GRADIO_AUTH=admin:strongpassword123
```

Or implement custom authentication in your reverse proxy.

## Scaling

### Horizontal Scaling

Run multiple instances behind a load balancer:

```bash
# docker-compose-scale.yml
version: '3.8'
services:
  app:
    image: text-to-image-generator:latest
    deploy:
      replicas: 3
    # ... rest of configuration
```

Start scaled deployment:
```bash
docker-compose -f docker-compose-scale.yml up -d --scale app=3
```

### Vertical Scaling

Allocate more resources to containers:

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Kubernetes Deployment

For Kubernetes, create deployment manifests:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-to-image-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: text-to-image
  template:
    metadata:
      labels:
        app: text-to-image
    spec:
      containers:
      - name: app
        image: text-to-image-generator:latest
        ports:
        - containerPort: 7860
        envFrom:
        - secretRef:
            name: app-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs app

# Common issues:
# 1. Missing API keys in .env
# 2. Port 7860 already in use
# 3. Insufficient disk space
```

### High Memory Usage

```bash
# Check container stats
docker stats text-to-image-app

# Solutions:
# 1. Increase memory limit in docker-compose.yml
# 2. Use lighter models (SD-Turbo instead of SDXL)
# 3. Reduce concurrent requests
```

### Health Check Failing

```bash
# Test health endpoint manually
curl http://localhost:7860/health

# Common causes:
# 1. Application not fully started (wait 60s)
# 2. High resource usage (check metrics)
# 3. Backend connectivity issues
```

### Rate Limiting Too Aggressive

```bash
# Adjust limits in .env
RATE_LIMIT_REQUESTS=200  # Increase
RATE_LIMIT_WINDOW=60

# Or disable temporarily
ENABLE_RATE_LIMITING=false

# Restart container
docker-compose restart app
```

### Slow Image Generation

```bash
# Use faster backends:
DEFAULT_BACKEND=replicate  # Usually faster than HuggingFace

# Or use local with turbo models:
DEFAULT_BACKEND=local
LOCAL_MODEL=stabilityai/sd-turbo

# Reduce inference steps (lower quality, faster):
# Configure in UI or via API
```

### Docker Build Fails

```bash
# Clean build
docker-compose build --no-cache

# Check Docker disk space
docker system df

# Prune if needed
docker system prune -a
```

## Production Checklist

Before going live:

- [ ] Set `PRODUCTION_MODE=true`
- [ ] Configure proper API keys
- [ ] Enable rate limiting
- [ ] Set up HTTPS/TLS
- [ ] Configure health checks
- [ ] Set up monitoring/alerting
- [ ] Test backup and restore
- [ ] Document runbook procedures
- [ ] Configure log aggregation
- [ ] Set up auto-scaling (if needed)
- [ ] Perform load testing
- [ ] Review security settings
- [ ] Set up CI/CD pipeline

## Support

For issues and questions:
- GitHub Issues: https://github.com/bailey-steve/text-to-image-generator/issues
- Documentation: See README.md

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Gradio Documentation](https://gradio.app/docs/)
- [HuggingFace Documentation](https://huggingface.co/docs)
