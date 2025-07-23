# 🚀 Deployment Guide

Complete guide for deploying the Smart Pill Recognition System in production environments.

## 📋 Table of Contents

- [🐳 Docker Deployment](#-docker-deployment)
- [☁️ Cloud Platforms](#️-cloud-platforms)
- [🖥️ Local Server](#️-local-server)
- [⚖️ Load Balancing](#️-load-balancing)
- [📊 Monitoring](#-monitoring)
- [🔒 Security](#-security)

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build the image
docker build -t pill-recognition:latest .

# Run with GPU support
docker run --gpus all -p 8501:8501 pill-recognition:latest

# Run CPU-only
docker run -p 8501:8501 pill-recognition:cpu
```

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  pill-recognition:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/checkpoints/best_model.pth
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - pill-recognition
```

### Production Dockerfile

```dockerfile
# Multi-stage build for production
FROM python:3.10-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt

FROM python:3.10-slim

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## ☁️ Cloud Platforms

### AWS Deployment

#### EC2 with GPU

```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type g4dn.xlarge \
    --key-name my-key-pair \
    --security-groups pill-recognition-sg

# Install dependencies
ssh -i my-key.pem ubuntu@ec2-instance
sudo apt update
sudo apt install docker.io nvidia-docker2

# Deploy application
git clone https://github.com/HoangThinh2024/DoAnDLL.git
cd DoAnDLL
docker-compose up -d
```

#### ECS Fargate

```json
{
  "family": "pill-recognition",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "pill-recognition",
      "image": "your-account.dkr.ecr.region.amazonaws.com/pill-recognition:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/checkpoints/best_model.pth"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/pill-recognition",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/pill-recognition

# Deploy to Cloud Run
gcloud run deploy pill-recognition \
    --image gcr.io/PROJECT_ID/pill-recognition \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2
```

#### GKE Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pill-recognition
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pill-recognition
  template:
    metadata:
      labels:
        app: pill-recognition
    spec:
      containers:
      - name: pill-recognition
        image: gcr.io/PROJECT_ID/pill-recognition:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_PATH
          value: "/app/checkpoints/best_model.pth"
---
apiVersion: v1
kind: Service
metadata:
  name: pill-recognition-service
spec:
  selector:
    app: pill-recognition
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

### Microsoft Azure

#### Container Instances

```bash
# Create resource group
az group create --name pill-recognition-rg --location eastus

# Deploy container
az container create \
    --resource-group pill-recognition-rg \
    --name pill-recognition \
    --image youracr.azurecr.io/pill-recognition:latest \
    --cpu 2 \
    --memory 4 \
    --gpu-count 1 \
    --gpu-sku V100 \
    --ports 8501 \
    --dns-name-label pill-recognition-app
```

## 🖥️ Local Server

### Ubuntu Server Setup

```bash
# 1. Install dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv nginx certbot

# 2. Clone and setup application
git clone https://github.com/HoangThinh2024/DoAnDLL.git
cd DoAnDLL
./bin/pill-setup

# 3. Create systemd service
sudo tee /etc/systemd/system/pill-recognition.service > /dev/null <<EOF
[Unit]
Description=Smart Pill Recognition System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/DoAnDLL
Environment=PATH=/home/ubuntu/DoAnDLL/.venv/bin
ExecStart=/home/ubuntu/DoAnDLL/.venv/bin/streamlit run app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 4. Enable and start service
sudo systemctl enable pill-recognition
sudo systemctl start pill-recognition

# 5. Configure Nginx reverse proxy
sudo tee /etc/nginx/sites-available/pill-recognition > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/pill-recognition /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 6. Setup SSL with Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

## ⚖️ Load Balancing

### Nginx Load Balancer

```nginx
upstream pill_recognition {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://pill_recognition;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### HAProxy Configuration

```
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend pill_recognition_frontend
    bind *:80
    default_backend pill_recognition_backend

backend pill_recognition_backend
    balance roundrobin
    server web1 127.0.0.1:8501 check
    server web2 127.0.0.1:8502 check
    server web3 127.0.0.1:8503 check
```

## 📊 Monitoring

### Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### Application Metrics

```python
# Add to your application
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('pill_recognition_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('pill_recognition_request_duration_seconds', 'Request latency')

@REQUEST_LATENCY.time()
def predict_pill(image, text):
    REQUEST_COUNT.inc()
    # Your prediction logic here
    pass
```

## 🔒 Security

### Security Checklist

- [ ] **HTTPS**: SSL/TLS certificates properly configured
- [ ] **Authentication**: API key or OAuth2 authentication
- [ ] **Rate Limiting**: Prevent abuse and DoS attacks
- [ ] **Input Validation**: Sanitize all user inputs
- [ ] **File Upload Security**: Validate file types and sizes
- [ ] **Environment Variables**: Store secrets securely
- [ ] **Network Security**: Firewall and VPC configuration
- [ ] **Monitoring**: Log all access and errors

### Rate Limiting with Redis

```python
import redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

redis_client = redis.Redis(host='localhost', port=6379)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)

@app.route('/predict')
@limiter.limit("10 per minute")
def predict():
    # Your prediction logic
    pass
```

### Environment Configuration

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
MODEL_PATH=/path/to/model
API_KEY=your-api-key
```

## 🔧 Performance Optimization

### CPU Optimization

```python
# In your Streamlit app
import streamlit as st

@st.cache_resource
def load_model():
    # Model loading code
    pass

@st.cache_data
def preprocess_image(image_bytes):
    # Image preprocessing
    pass
```

### GPU Memory Management

```python
import torch

# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use mixed precision training
from torch.cuda.amp import autocast

with autocast():
    output = model(input_tensor)
```

## 📝 Deployment Checklist

- [ ] **Environment Setup**: All dependencies installed
- [ ] **Model Files**: Model checkpoints available
- [ ] **Configuration**: All config files properly set
- [ ] **SSL Certificates**: HTTPS properly configured
- [ ] **Monitoring**: Logging and metrics in place
- [ ] **Backup**: Database and model backups configured
- [ ] **Load Testing**: Performance validated under load
- [ ] **Security**: Security measures implemented
- [ ] **Documentation**: Deployment documented

## 🆘 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use CPU inference
2. **CUDA Errors**: Check GPU driver and CUDA version compatibility
3. **Port Conflicts**: Use different ports or kill conflicting processes
4. **Permission Denied**: Check file permissions and user access
5. **SSL Issues**: Verify certificate configuration

### Debug Commands

```bash
# Check application logs
docker logs pill-recognition

# Monitor GPU usage
nvidia-smi -l 1

# Check disk space
df -h

# Monitor network connections
netstat -tlnp | grep :8501

# Test API endpoint
curl -X POST http://localhost:8501/predict \
     -F "image=@test_image.jpg" \
     -F "text=ADVIL 200"
```

---

For more detailed deployment scenarios, please refer to the platform-specific documentation or create an issue for assistance.