# Final Project Setup Guide

## Overview
This guide will help you set up the complete development environment for the final web crawling platform project. The project consists of multiple services that need to be coordinated and configured properly.

## Architecture Overview

The final project consists of four main services:

1. **Backend API** (FastAPI + PostgreSQL): REST API for managing crawls and data
2. **Crawler Service** (Python + Crawlee): Asynchronous web crawling engine
3. **Frontend** (Next.js + React): Web interface for monitoring and management
4. **AI Service** (Optional): Dedicated AI processing service

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows 10/11 with WSL2
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 20GB+ free space
- **Network**: Stable internet connection

### Required Software
- **Docker & Docker Compose**: For containerized development
- **Git**: Version control
- **Python 3.9+**: Backend and crawler services
- **Node.js 18+**: Frontend development
- **PostgreSQL 13+**: Database (via Docker)
- **Redis 6+**: Caching and message broker (via Docker)

### API Keys and Services
- **OpenAI API Key**: For AI content processing
- **GitHub Account**: For version control and CI/CD
- **Docker Hub Account**: For container registry (optional)

## Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the project repository
git clone <your-project-repository-url>
cd final-project

# Create necessary directories
mkdir -p logs backups data/redis data/postgres

# Set proper permissions
chmod 755 scripts/*.sh
```

### Step 2: Environment Configuration

#### Create Environment Files

```bash
# Backend environment
cat > backend/.env << EOF
# Database
DATABASE_URL=postgresql+asyncpg://crawler_user:crawler_pass@localhost:5432/crawler_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key
API_KEY=your-api-key-for-external-access

# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]

# Crawler Settings
MAX_CONCURRENT_CRAWLS=5
CRAWL_TIMEOUT=30
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# File Storage
UPLOAD_DIR=/app/uploads
EXPORT_DIR=/app/exports
EOF

# Frontend environment
cat > frontend/.env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_APP_ENV=development
NEXT_PUBLIC_VERSION=1.0.0

# Analytics (optional)
NEXT_PUBLIC_GA_TRACKING_ID=your-ga-tracking-id
EOF

# Crawler environment
cat > crawler/.env << EOF
DATABASE_URL=postgresql+asyncpg://crawler_user:crawler_pass@localhost:5432/crawler_db
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Crawler Configuration
MAX_REQUESTS_PER_CRAWL=1000
CONCURRENT_REQUESTS=10
CRAWL_DELAY=1.0
USER_AGENT=CrawlerPlatform/1.0

# Storage
DATA_DIR=/app/data
LOG_DIR=/app/logs

# Monitoring
PROMETHEUS_PORT=8001
EOF
```

#### Configure Docker Environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: crawler_db
      POSTGRES_USER: crawler_user
      POSTGRES_PASSWORD: crawler_pass
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U crawler_user -d crawler_db"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis Cache/Message Broker
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql+asyncpg://crawler_user:crawler_pass@postgres:5432/crawler_db
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Crawler Service
  crawler:
    build:
      context: ./crawler
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql+asyncpg://crawler_user:crawler_pass@postgres:5432/crawler_db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./crawler:/app
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - backend
      - redis

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
```

### Step 3: Database Initialization

#### Create Database Schema

```sql
-- scripts/init-db.sql
-- Database initialization script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- Create indexes for better performance
-- (Additional schema will be created by Alembic migrations)
```

#### Initialize Alembic for Migrations

```bash
# Backend setup
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Initialize Alembic (if not already done)
alembic init migrations

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Run migrations
alembic upgrade head
```

### Step 4: Build and Start Services

#### Using Docker Compose (Recommended)

```bash
# Build all services
docker-compose build

# Start services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f
```

#### Manual Development Setup (Alternative)

```bash
# Terminal 1: Start database and Redis
docker-compose up postgres redis -d

# Terminal 2: Backend API
cd backend
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd frontend
npm install
npm run dev

# Terminal 4: Crawler Service
cd crawler
pip install -r requirements.txt
python -m crawlers.main
```

### Step 5: Verify Installation

#### Health Checks

```bash
# Check database connection
docker-compose exec postgres pg_isready -U crawler_user -d crawler_db

# Check Redis connection
docker-compose exec redis redis-cli ping

# Check backend API
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000
```

#### Run Tests

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app --cov-report=html

# Frontend tests
cd frontend
npm test

# Crawler tests
cd crawler
python -m pytest tests/ -v
```

#### Access the Application

- **Frontend Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Database Admin** (if configured): http://localhost:8080

## Configuration Details

### Backend Configuration

```python
# backend/app/core/config.py
from pydantic import BaseSettings, validator
from typing import List, Optional
import secrets

class Settings(BaseSettings):
    # Application
    app_name: str = "Crawler Platform API"
    version: str = "1.0.0"
    debug: bool = False
    secret_key: str = secrets.token_urlsafe(32)

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Security
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # CORS
    cors_origins: List[str] = ["http://localhost:3000"]

    # Crawler Settings
    max_concurrent_crawls: int = 5
    crawl_timeout: int = 30
    rate_limit_requests_per_minute: int = 60

    # AI Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000

    # File Storage
    upload_dir: str = "/app/uploads"
    export_dir: str = "/app/exports"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Crawler Configuration

```python
# crawler/config/settings.py
from pydantic import BaseSettings
from typing import Optional

class CrawlerSettings(BaseSettings):
    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Crawler
    max_requests_per_crawl: int = 1000
    concurrent_requests: int = 10
    crawl_delay: float = 1.0
    user_agent: str = "CrawlerPlatform/1.0"
    respect_robots_txt: bool = True

    # AI
    openai_api_key: Optional[str] = None

    # Storage
    data_dir: str = "/app/data"
    log_dir: str = "/app/logs"

    # Monitoring
    prometheus_port: int = 8001
    enable_metrics: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

crawler_settings = CrawlerSettings()
```

### Frontend Configuration

```typescript
// frontend/config/app.ts
export const appConfig = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  appEnv: process.env.NEXT_PUBLIC_APP_ENV || 'development',
  version: process.env.NEXT_PUBLIC_VERSION || '1.0.0',

  // Feature flags
  features: {
    aiProcessing: true,
    realTimeUpdates: true,
    advancedSearch: true,
    exportData: true,
  },

  // UI Configuration
  ui: {
    theme: 'light',
    language: 'en',
    dateFormat: 'YYYY-MM-DD',
    itemsPerPage: 20,
  },

  // API endpoints
  endpoints: {
    crawls: '/api/crawls',
    search: '/api/search',
    analytics: '/api/analytics',
    health: '/health',
  }
}
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up postgres -d
```

#### Redis Connection Issues
```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis

# Restart Redis
docker-compose restart redis
```

#### Backend Startup Issues
```bash
# Check backend logs
docker-compose logs backend

# Check if dependencies are installed
docker-compose exec backend pip list

# Restart backend
docker-compose restart backend
```

#### Frontend Build Issues
```bash
# Clear Next.js cache
cd frontend
rm -rf .next node_modules
npm install
npm run build
```

#### Port Conflicts
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8000
lsof -i :5432

# Change ports in docker-compose.yml if needed
```

### Performance Tuning

#### Database Optimization
```sql
-- Check database performance
SELECT * FROM pg_stat_activity;
SELECT * FROM pg_stat_user_tables;
ANALYZE; -- Update statistics
```

#### Memory Usage
```bash
# Monitor container memory usage
docker stats

# Adjust Docker memory limits in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

#### Scaling Services
```bash
# Scale crawler service
docker-compose up -d --scale crawler=3

# Add more backend instances
docker-compose up -d --scale backend=2
```

## Development Workflow

### Code Quality Tools

```bash
# Backend linting
cd backend
black . --check
isort . --check-only
flake8 .
mypy .

# Frontend linting
cd frontend
npm run lint
npm run type-check

# Run all tests
docker-compose exec backend pytest
docker-compose exec frontend npm test
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push and create PR
git push origin feature/new-feature
```

### Deployment

#### Staging Deployment
```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run integration tests
npm run test:integration
```

#### Production Deployment
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Run health checks
curl -f https://your-domain.com/health
```

## Security Considerations

### Environment Variables
- Never commit secrets to version control
- Use different secrets for each environment
- Rotate API keys regularly

### Network Security
```yaml
# Add to docker-compose.yml for production
services:
  backend:
    networks:
      - internal
    # No external port exposure

  frontend:
    networks:
      - internal
      - external
```

### Data Protection
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement proper authentication and authorization
- Regular security audits and updates

## Monitoring and Logging

### Application Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend

# Follow logs with timestamps
docker-compose logs -f --timestamps
```

### Health Monitoring
```bash
# Check all service health
curl http://localhost:8000/health
curl http://localhost:3000/api/health

# Database health
docker-compose exec postgres pg_isready -U crawler_user -d crawler_db
```

### Performance Monitoring
```bash
# Check resource usage
docker stats

# Database performance
docker-compose exec postgres psql -U crawler_user -d crawler_db -c "SELECT * FROM pg_stat_activity;"
```

## Next Steps

Once your environment is set up:

1. **Explore the codebase** - Familiarize yourself with the project structure
2. **Run the tests** - Ensure everything is working correctly
3. **Start development** - Begin implementing features according to the project roadmap
4. **Set up CI/CD** - Configure automated testing and deployment
5. **Monitor performance** - Use the built-in monitoring tools to track system health

## Support

If you encounter issues during setup:

1. Check the troubleshooting section above
2. Review the application logs for error messages
3. Verify all prerequisites are installed correctly
4. Check the project documentation and README files
5. Create an issue in the project repository with detailed information

Happy coding! 🚀