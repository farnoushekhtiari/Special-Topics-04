# Workshop 06: Pipeline Monitoring and Analytics

## Overview
This workshop focuses on comprehensive monitoring, analytics, and observability for crawling pipelines. You'll implement real-time monitoring, performance analytics, alerting systems, and visualization dashboards to ensure pipeline reliability and optimize performance.

## Prerequisites
- Completed [Pipeline Monitoring Tutorial](../tutorials/06-pipeline-monitoring.md)
- Knowledge of monitoring concepts and metrics
- Understanding of data visualization and analytics

## Learning Objectives
By the end of this workshop, you will be able to:
- Implement comprehensive pipeline monitoring and alerting
- Create performance analytics and bottleneck detection
- Build real-time dashboards and visualization
- Set up automated anomaly detection and alerting
- Implement pipeline health checks and recovery mechanisms

## Workshop Structure

### Part 1: Monitoring Infrastructure

#### Step 1: Create Metrics Collection System

```python
# monitoring/metrics_collector.py
import asyncio
import time
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import statistics


@dataclass
class MetricPoint:
    """Individual metric measurement"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical summary of a metric"""
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    last_value: float
    last_updated: datetime


class MetricsCollector:
    """Central metrics collection system"""

    def __init__(self, retention_period: int = 3600):  # 1 hour default
        self.metrics: Dict[str, deque] = {}
        self.retention_period = retention_period
        self.lock = asyncio.Lock()
        self._collection_task: Optional[asyncio.Task] = None
        self.custom_collectors: List[Callable] = []

    async def start_collection(self, interval: int = 60):
        """Start periodic metrics collection"""
        self._collection_task = asyncio.create_task(self._collection_loop(interval))

    async def stop_collection(self):
        """Stop metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

    async def _collection_loop(self, interval: int):
        """Main collection loop"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self._collect_system_metrics()
                await self._collect_custom_metrics()
                await self._cleanup_old_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics collection error: {e}")

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("system.cpu.percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric("system.memory.percent", memory.percent)
            await self.record_metric("system.memory.used_bytes", memory.used)
            await self.record_metric("system.memory.available_bytes", memory.available)

            # Disk metrics
            disk = psutil.disk_usage('/')
            await self.record_metric("system.disk.percent", disk.percent)
            await self.record_metric("system.disk.used_bytes", disk.used)
            await self.record_metric("system.disk.free_bytes", disk.free)

            # Network metrics
            network = psutil.net_io_counters()
            await self.record_metric("system.network.bytes_sent", network.bytes_sent)
            await self.record_metric("system.network.bytes_recv", network.bytes_recv)

        except Exception as e:
            print(f"System metrics collection error: {e}")

    async def _collect_custom_metrics(self):
        """Collect custom metrics from registered collectors"""
        for collector in self.custom_collectors:
            try:
                await collector(self)
            except Exception as e:
                print(f"Custom metrics collection error: {e}")

    def add_custom_collector(self, collector: Callable):
        """Add a custom metrics collector"""
        self.custom_collectors.append(collector)

    async def record_metric(self, name: str, value: float,
                           tags: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a metric measurement"""
        async with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque()

            point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {}
            )

            self.metrics[name].append(point)

    async def record_counter(self, name: str, increment: float = 1.0,
                            tags: Optional[Dict[str, str]] = None):
        """Record a counter metric (incrementing value)"""
        current_value = await self.get_metric_value(name, tags)
        new_value = (current_value or 0) + increment
        await self.record_metric(name, new_value, tags)

    async def record_gauge(self, name: str, value: float,
                          tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric (point-in-time value)"""
        await self.record_metric(name, value, tags)

    async def record_histogram(self, name: str, value: float,
                              tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        await self.record_metric(name, value, tags, {"type": "histogram"})

    async def get_metric_value(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get the latest value for a metric"""
        if name not in self.metrics:
            return None

        # Filter by tags if provided
        points = list(self.metrics[name])
        if tags:
            points = [p for p in points if all(p.tags.get(k) == v for k, v in tags.items())]

        if not points:
            return None

        # Return most recent value
        return max(points, key=lambda p: p.timestamp).value

    async def get_metric_stats(self, name: str, time_range: int = 3600) -> Optional[MetricStats]:
        """Get statistical summary for a metric"""
        if name not in self.metrics:
            return None

        cutoff_time = datetime.now() - timedelta(seconds=time_range)
        points = [p for p in self.metrics[name] if p.timestamp > cutoff_time]

        if not points:
            return None

        values = [p.value for p in points]
        last_point = max(points, key=lambda p: p.timestamp)

        return MetricStats(
            name=name,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            p95_value=statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            p99_value=statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
            last_value=last_point.value,
            last_updated=last_point.timestamp
        )

    async def get_all_metrics(self) -> Dict[str, MetricStats]:
        """Get stats for all metrics"""
        stats = {}
        for name in self.metrics.keys():
            metric_stats = await self.get_metric_stats(name)
            if metric_stats:
                stats[name] = metric_stats
        return stats

    async def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        async with self.lock:
            cutoff_time = datetime.now() - timedelta(seconds=self.retention_period)

            for name, points in self.metrics.items():
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()

                # Remove empty metric queues
                if not points:
                    del self.metrics[name]

    async def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        all_stats = await self.get_all_metrics()

        if format == "json":
            return json.dumps({
                name: {
                    'count': stats.count,
                    'min': stats.min_value,
                    'max': stats.max_value,
                    'avg': stats.avg_value,
                    'median': stats.median_value,
                    'p95': stats.p95_value,
                    'p99': stats.p99_value,
                    'last': stats.last_value,
                    'last_updated': stats.last_updated.isoformat()
                }
                for name, stats in all_stats.items()
            }, indent=2)

        elif format == "prometheus":
            lines = []
            for name, stats in all_stats.items():
                # Convert metric name to Prometheus format
                prom_name = name.replace('.', '_').replace('-', '_')

                lines.extend([
                    f"# HELP {prom_name} {name}",
                    f"# TYPE {prom_name} gauge",
                    f"{prom_name}{{stat=\"count\"}} {stats.count}",
                    f"{prom_name}{{stat=\"min\"}} {stats.min_value}",
                    f"{prom_name}{{stat=\"max\"}} {stats.max_value}",
                    f"{prom_name}{{stat=\"avg\"}} {stats.avg_value}",
                    f"{prom_name}{{stat=\"median\"}} {stats.median_value}",
                    f"{prom_name}{{stat=\"p95\"}} {stats.p95_value}",
                    f"{prom_name}{{stat=\"p99\"}} {stats.p99_value}",
                    f"{prom_name}{{stat=\"last\"}} {stats.last_value}",
                ])
            return '\n'.join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")


class PipelineMetricsCollector:
    """Metrics collector specifically for crawling pipelines"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector

    async def record_crawler_start(self, session_id: str):
        """Record crawler session start"""
        await self.metrics.record_counter("pipeline.sessions.started")
        await self.metrics.record_gauge("pipeline.sessions.active", 1,
                                       tags={"operation": "increment"})

    async def record_crawler_end(self, session_id: str, duration: float,
                                urls_processed: int, articles_found: int):
        """Record crawler session end"""
        await self.metrics.record_counter("pipeline.sessions.completed")
        await self.metrics.record_gauge("pipeline.sessions.active", -1,
                                       tags={"operation": "decrement"})

        await self.metrics.record_histogram("pipeline.session.duration", duration,
                                          tags={"session_id": session_id})
        await self.metrics.record_histogram("pipeline.session.urls_processed", urls_processed,
                                          tags={"session_id": session_id})
        await self.metrics.record_histogram("pipeline.session.articles_found", articles_found,
                                          tags={"session_id": session_id})

        # Calculate efficiency metrics
        if duration > 0:
            urls_per_second = urls_processed / duration
            articles_per_second = articles_found / duration

            await self.metrics.record_gauge("pipeline.performance.urls_per_second", urls_per_second)
            await self.metrics.record_gauge("pipeline.performance.articles_per_second", articles_per_second)

    async def record_url_processed(self, url: str, success: bool, response_time: float,
                                  status_code: Optional[int] = None):
        """Record URL processing metrics"""
        await self.metrics.record_counter("pipeline.urls.total")

        if success:
            await self.metrics.record_counter("pipeline.urls.success")
        else:
            await self.metrics.record_counter("pipeline.urls.failed")

        await self.metrics.record_histogram("pipeline.urls.response_time", response_time)

        if status_code:
            await self.metrics.record_counter(f"pipeline.urls.status.{status_code}")

    async def record_article_stored(self, article_id: str, word_count: int, processing_time: float):
        """Record article storage metrics"""
        await self.metrics.record_counter("pipeline.articles.stored")
        await self.metrics.record_histogram("pipeline.articles.word_count", word_count)
        await self.metrics.record_histogram("pipeline.articles.processing_time", processing_time)

    async def record_duplicate_detected(self, confidence: float):
        """Record duplicate detection metrics"""
        await self.metrics.record_counter("pipeline.duplicates.detected")
        await self.metrics.record_histogram("pipeline.duplicates.confidence", confidence)

    async def record_search_query(self, query: str, result_count: int, response_time: float):
        """Record search query metrics"""
        await self.metrics.record_counter("pipeline.search.queries")
        await self.metrics.record_histogram("pipeline.search.response_time", response_time)
        await self.metrics.record_histogram("pipeline.search.result_count", result_count)

        # Track query length
        await self.metrics.record_histogram("pipeline.search.query_length", len(query))

    async def record_error(self, error_type: str, error_message: str):
        """Record error metrics"""
        await self.metrics.record_counter(f"pipeline.errors.{error_type}")
        await self.metrics.record_counter("pipeline.errors.total")

    async def record_queue_size(self, queue_size: int):
        """Record queue size metrics"""
        await self.metrics.record_gauge("pipeline.queue.size", queue_size)

    async def record_database_operation(self, operation: str, table: str, duration: float):
        """Record database operation metrics"""
        await self.metrics.record_histogram(f"pipeline.db.{operation}.{table}", duration)
        await self.metrics.record_histogram("pipeline.db.duration", duration,
                                          tags={"operation": operation, "table": table})
```

#### Step 2: Create Alerting System

```python
# monitoring/alert_system.py
import asyncio
import smtplib
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp


@dataclass
class AlertRule:
    """Configuration for an alert rule"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 90", "< 10", "== 0"
    threshold: float
    duration: int  # seconds - how long condition must be true
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    cooldown: int = 300  # seconds between alerts
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance"""
    rule_name: str
    severity: str
    description: str
    triggered_at: datetime
    value: float
    threshold: float
    condition: str
    resolved_at: Optional[datetime] = None


class AlertManager:
    """Alert management system"""

    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notifiers: List[Callable] = []

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules.append(rule)

    def add_notifier(self, notifier: Callable):
        """Add an alert notifier"""
        self.notifiers.append(notifier)

    async def check_alerts(self):
        """Check all alert rules and trigger alerts"""
        current_time = datetime.now()

        for rule in self.alert_rules:
            try:
                # Get metric stats for the specified duration
                stats = await self.metrics.get_metric_stats(rule.metric_name, rule.duration)

                if not stats:
                    continue

                # Evaluate condition
                alert_triggered = self._evaluate_condition(stats.last_value, rule.condition, rule.threshold)

                alert_key = f"{rule.name}_{rule.metric_name}"

                if alert_triggered:
                    # Check if alert is already active
                    if alert_key not in self.active_alerts:
                        # Check cooldown
                        last_alert = self._get_last_alert_for_rule(rule.name)
                        if last_alert and (current_time - last_alert.triggered_at).seconds < rule.cooldown:
                            continue

                        # Create new alert
                        alert = Alert(
                            rule_name=rule.name,
                            severity=rule.severity,
                            description=rule.description,
                            triggered_at=current_time,
                            value=stats.last_value,
                            threshold=rule.threshold,
                            condition=rule.condition
                        )

                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)

                        # Notify
                        await self._notify_alert(alert)

                else:
                    # Resolve alert if it was active
                    if alert_key in self.active_alerts:
                        alert = self.active_alerts[alert_key]
                        alert.resolved_at = current_time
                        del self.active_alerts[alert_key]

            except Exception as e:
                print(f"Error checking alert rule {rule.name}: {e}")

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith('>'):
                return value > threshold
            elif condition.startswith('>='):
                return value >= threshold
            elif condition.startswith('<'):
                return value < threshold
            elif condition.startswith('<='):
                return value <= threshold
            elif condition.startswith('=='):
                return abs(value - threshold) < 0.001  # Floating point comparison
            elif condition.startswith('!='):
                return abs(value - threshold) >= 0.001
            else:
                return False
        except (ValueError, TypeError):
            return False

    def _get_last_alert_for_rule(self, rule_name: str) -> Optional[Alert]:
        """Get the last alert for a specific rule"""
        rule_alerts = [a for a in self.alert_history if a.rule_name == rule_name]
        return max(rule_alerts, key=lambda a: a.triggered_at) if rule_alerts else None

    async def _notify_alert(self, alert: Alert):
        """Notify all notifiers about the alert"""
        for notifier in self.notifiers:
            try:
                await notifier(alert)
            except Exception as e:
                print(f"Alert notification failed: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at > cutoff_time]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_1h = now - timedelta(hours=1)

        alerts_24h = [a for a in self.alert_history if a.triggered_at > last_24h]
        alerts_1h = [a for a in self.alert_history if a.triggered_at > last_1h]

        severity_counts = {}
        for alert in alerts_24h:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        return {
            'active_alerts': len(self.active_alerts),
            'alerts_last_24h': len(alerts_24h),
            'alerts_last_1h': len(alerts_1h),
            'severity_breakdown': severity_counts,
            'most_recent_alert': max(self.alert_history, key=lambda a: a.triggered_at) if self.alert_history else None
        }


class EmailNotifier:
    """Email alert notifier"""

    def __init__(self, smtp_server: str, smtp_port: int = 587,
                 username: str = None, password: str = None,
                 recipients: List[str] = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients or []

    async def __call__(self, alert: Alert):
        """Send email notification for alert"""
        if not self.recipients:
            return

        subject = f"[{alert.severity.upper()}] Crawling Pipeline Alert: {alert.rule_name}"

        body = f"""
Crawling Pipeline Alert

Severity: {alert.severity.upper()}
Rule: {alert.rule_name}
Description: {alert.description}

Details:
- Current Value: {alert.value}
- Threshold: {alert.threshold}
- Condition: {alert.condition}
- Triggered: {alert.triggered_at}

Please check the pipeline monitoring dashboard for more details.
        """

        # Send email asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_email_sync, subject, body)

    def _send_email_sync(self, subject: str, body: str):
        """Send email synchronously"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username or 'alerts@crawler.com'
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.sendmail(msg['From'], self.recipients, msg.as_string())
            server.quit()

        except Exception as e:
            print(f"Failed to send email alert: {e}")


class SlackNotifier:
    """Slack alert notifier"""

    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel

    async def __call__(self, alert: Alert):
        """Send Slack notification for alert"""
        severity_colors = {
            'low': 'good',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'danger'
        }

        payload = {
            'attachments': [{
                'color': severity_colors.get(alert.severity, 'danger'),
                'title': f"Crawling Pipeline Alert: {alert.rule_name}",
                'text': alert.description,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.upper(), 'short': True},
                    {'title': 'Current Value', 'value': f"{alert.value}", 'short': True},
                    {'title': 'Threshold', 'value': f"{alert.threshold}", 'short': True},
                    {'title': 'Condition', 'value': alert.condition, 'short': True},
                    {'title': 'Triggered', 'value': alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                ]
            }]
        }

        if self.channel:
            payload['channel'] = self.channel

        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    print(f"Slack notification failed: {response.status}")


class WebhookNotifier:
    """Generic webhook notifier"""

    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def __call__(self, alert: Alert):
        """Send webhook notification for alert"""
        payload = {
            'alert': {
                'rule_name': alert.rule_name,
                'severity': alert.severity,
                'description': alert.description,
                'triggered_at': alert.triggered_at.isoformat(),
                'value': alert.value,
                'threshold': alert.threshold,
                'condition': alert.condition
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload, headers=self.headers) as response:
                if response.status not in (200, 201, 202):
                    print(f"Webhook notification failed: {response.status}")


def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules for crawling pipelines"""
    return [
        AlertRule(
            name="high_cpu_usage",
            metric_name="system.cpu.percent",
            condition="> 90",
            threshold=90.0,
            duration=300,  # 5 minutes
            severity="high",
            description="CPU usage is above 90% for 5 minutes",
            cooldown=600
        ),
        AlertRule(
            name="high_memory_usage",
            metric_name="system.memory.percent",
            condition="> 85",
            threshold=85.0,
            duration=300,
            severity="high",
            description="Memory usage is above 85% for 5 minutes",
            cooldown=600
        ),
        AlertRule(
            name="low_disk_space",
            metric_name="system.disk.percent",
            condition="> 90",
            threshold=90.0,
            duration=3600,  # 1 hour
            severity="critical",
            description="Disk usage is above 90% for 1 hour",
            cooldown=3600
        ),
        AlertRule(
            name="crawler_failures",
            metric_name="pipeline.urls.failed",
            condition="> 50",
            threshold=50.0,
            duration=600,  # 10 minutes
            severity="medium",
            description="More than 50 URL processing failures in 10 minutes",
            cooldown=1800
        ),
        AlertRule(
            name="slow_responses",
            metric_name="pipeline.urls.response_time",
            condition="> 30",
            threshold=30.0,
            duration=600,
            severity="medium",
            description="Average response time above 30 seconds for 10 minutes",
            cooldown=1800
        ),
        AlertRule(
            name="queue_backup",
            metric_name="pipeline.queue.size",
            condition="> 1000",
            threshold=1000.0,
            duration=300,
            severity="high",
            description="URL queue size exceeds 1000 for 5 minutes",
            cooldown=900
        )
    ]
```

### Part 2: Dashboard and Visualization

#### Step 1: Create Monitoring Dashboard

```python
# dashboard/monitoring_dashboard.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uvicorn

from monitoring.metrics_collector import MetricsCollector, PipelineMetricsCollector
from monitoring.alert_system import AlertManager


class MonitoringDashboard:
    """Web-based monitoring dashboard"""

    def __init__(self, metrics_collector: MetricsCollector,
                 alert_manager: AlertManager,
                 pipeline_metrics: PipelineMetricsCollector):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.pipeline_metrics = pipeline_metrics

        self.app = FastAPI(title="Crawling Pipeline Monitor", version="1.0.0")

        # Setup static files and templates
        self._setup_static_files()
        self._setup_routes()

    def _setup_static_files(self):
        """Setup static file serving"""
        static_path = Path(__file__).parent / "static"
        static_path.mkdir(exist_ok=True)

        self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

        # Create basic CSS and JS files
        self._create_static_files(static_path)

    def _create_static_files(self, static_path: Path):
        """Create basic static files for the dashboard"""

        # CSS
        css_content = """
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; margin: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .alert-card { background: #ffeaa7; border-left: 5px solid #d63031; padding: 15px; margin: 10px 0; }
        .alert-critical { background: #fab1a0; border-left-color: #e17055; }
        .status-good { color: #00b894; }
        .status-warning { color: #fdcb6e; }
        .status-error { color: #e17055; }
        .chart-container { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        """

        with open(static_path / "styles.css", "w") as f:
            f.write(css_content)

        # JavaScript
        js_content = """
        // Auto-refresh functionality
        function autoRefresh() {
            setInterval(function() {
                location.reload();
            }, 30000); // Refresh every 30 seconds
        }

        // Start auto-refresh when page loads
        document.addEventListener('DOMContentLoaded', function() {
            autoRefresh();
        });

        // Chart.js integration (placeholder)
        function createChart(canvasId, data) {
            // Placeholder for chart creation
            console.log('Creating chart:', canvasId, data);
        }
        """

        with open(static_path / "scripts.js", "w") as f:
            f.write(js_content)

    def _setup_routes(self):
        """Setup dashboard routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page"""
            return await self._render_dashboard(request)

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics data"""
            try:
                all_metrics = await self.metrics.get_all_metrics()
                return {"metrics": all_metrics}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get current alerts"""
            try:
                active_alerts = self.alerts.get_active_alerts()
                alert_summary = self.alerts.get_alert_summary()

                return {
                    "active_alerts": active_alerts,
                    "summary": alert_summary
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Basic health checks
                metrics_count = len(await self.metrics.get_all_metrics())
                active_alerts = len(self.alerts.get_active_alerts())

                return {
                    "status": "healthy",
                    "timestamp": datetime.now(),
                    "metrics_count": metrics_count,
                    "active_alerts": active_alerts
                }
            except Exception:
                raise HTTPException(status_code=503, detail="Service unhealthy")

        @self.app.get("/metrics/{metric_name}")
        async def get_metric_detail(metric_name: str):
            """Get detailed information for a specific metric"""
            try:
                stats = await self.metrics.get_metric_stats(metric_name)
                if not stats:
                    raise HTTPException(status_code=404, detail="Metric not found")

                return stats.__dict__
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def _render_dashboard(self, request: Request) -> HTMLResponse:
        """Render the main dashboard HTML"""
        try:
            # Get current data
            metrics = await self.metrics.get_all_metrics()
            active_alerts = self.alerts.get_active_alerts()
            alert_summary = self.alerts.get_alert_summary()

            # Calculate status indicators
            system_status = self._calculate_system_status(metrics)
            pipeline_status = self._calculate_pipeline_status(metrics)

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Crawling Pipeline Monitor</title>
                <link rel="stylesheet" href="/static/styles.css">
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🕷️ Crawling Pipeline Monitor</h1>
                        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p>Status: <span class="status-{system_status['color']}">{system_status['text']}</span></p>
                    </div>

                    <!-- Active Alerts -->
                    {self._render_alerts_section(active_alerts)}

                    <!-- Key Metrics -->
                    <div class="metric-grid">
                        {self._render_metric_cards(metrics)}
                    </div>

                    <!-- Pipeline Performance -->
                    <div class="chart-container">
                        <h2>Pipeline Performance</h2>
                        <div class="metric-grid">
                            <div class="metric-card">
                                <h3>URLs Processed</h3>
                                <p class="metric-value">{pipeline_status.get('urls_processed', 0)}</p>
                            </div>
                            <div class="metric-card">
                                <h3>Articles Stored</h3>
                                <p class="metric-value">{pipeline_status.get('articles_stored', 0)}</p>
                            </div>
                            <div class="metric-card">
                                <h3>Success Rate</h3>
                                <p class="metric-value">{pipeline_status.get('success_rate', 0):.1%}</p>
                            </div>
                            <div class="metric-card">
                                <h3>Avg Response Time</h3>
                                <p class="metric-value">{pipeline_status.get('avg_response_time', 0):.2f}s</p>
                            </div>
                        </div>
                    </div>

                    <!-- System Resources -->
                    <div class="chart-container">
                        <h2>System Resources</h2>
                        <canvas id="systemChart" width="400" height="200"></canvas>
                    </div>
                </div>

                <script src="/static/scripts.js"></script>
                <script>
                    // Initialize charts
                    const systemData = {json.dumps(self._prepare_system_chart_data(metrics))};
                    createChart('systemChart', systemData);
                </script>
            </body>
            </html>
            """

            return HTMLResponse(content=html_content)

        except Exception as e:
            error_html = f"""
            <html>
            <body>
                <h1>Dashboard Error</h1>
                <p>Error loading dashboard: {str(e)}</p>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=500)

    def _render_alerts_section(self, alerts: List) -> str:
        """Render active alerts section"""
        if not alerts:
            return '<div class="metric-card"><h3>Active Alerts</h3><p>No active alerts</p></div>'

        alert_html = '<div class="metric-card"><h3>🚨 Active Alerts</h3>'
        for alert in alerts:
            severity_class = "alert-critical" if alert.severity == "critical" else "alert-card"
            alert_html += f'''
            <div class="{severity_class}">
                <strong>{alert.rule_name}</strong> ({alert.severity.upper()})
                <br>{alert.description}
                <br><small>Triggered: {alert.triggered_at.strftime('%H:%M:%S')}</small>
            </div>
            '''

        alert_html += '</div>'
        return alert_html

    def _render_metric_cards(self, metrics: Dict) -> str:
        """Render metric cards"""
        key_metrics = [
            ('system.cpu.percent', 'CPU Usage', '%'),
            ('system.memory.percent', 'Memory Usage', '%'),
            ('system.disk.percent', 'Disk Usage', '%'),
            ('pipeline.urls.total', 'Total URLs', ''),
            ('pipeline.articles.stored', 'Articles Stored', ''),
            ('pipeline.search.queries', 'Search Queries', ''),
        ]

        cards_html = ""
        for metric_name, display_name, unit in key_metrics:
            if metric_name in metrics:
                stats = metrics[metric_name]
                cards_html += f'''
                <div class="metric-card">
                    <h3>{display_name}</h3>
                    <p class="metric-value">{stats.last_value:.1f}{unit}</p>
                    <small>Avg: {stats.avg_value:.1f}{unit} | Max: {stats.max_value:.1f}{unit}</small>
                </div>
                '''

        return cards_html

    def _calculate_system_status(self, metrics: Dict) -> Dict[str, str]:
        """Calculate overall system status"""
        cpu_usage = metrics.get('system.cpu.percent')
        memory_usage = metrics.get('system.memory.percent')

        if not cpu_usage or not memory_usage:
            return {"text": "Unknown", "color": "warning"}

        cpu_val = cpu_usage.last_value
        mem_val = memory_usage.last_value

        if cpu_val > 90 or mem_val > 90:
            return {"text": "Critical", "color": "error"}
        elif cpu_val > 70 or mem_val > 80:
            return {"text": "Warning", "color": "warning"}
        else:
            return {"text": "Healthy", "color": "good"}

    def _calculate_pipeline_status(self, metrics: Dict) -> Dict[str, Any]:
        """Calculate pipeline performance status"""
        urls_total = metrics.get('pipeline.urls.total')
        urls_success = metrics.get('pipeline.urls.success')
        response_time = metrics.get('pipeline.urls.response_time')
        articles_stored = metrics.get('pipeline.articles.stored')

        status = {}

        if urls_total and urls_success:
            status['urls_processed'] = int(urls_total.last_value)
            status['success_rate'] = urls_success.last_value / urls_total.last_value

        if response_time:
            status['avg_response_time'] = response_time.avg_value

        if articles_stored:
            status['articles_stored'] = int(articles_stored.last_value)

        return status

    def _prepare_system_chart_data(self, metrics: Dict) -> Dict[str, Any]:
        """Prepare data for system resource chart"""
        # This would prepare Chart.js compatible data
        return {
            "labels": ["CPU", "Memory", "Disk"],
            "datasets": [{
                "label": "System Resources",
                "data": [
                    metrics.get('system.cpu.percent', {}).last_value if 'system.cpu.percent' in metrics else 0,
                    metrics.get('system.memory.percent', {}).last_value if 'system.memory.percent' in metrics else 0,
                    metrics.get('system.disk.percent', {}).last_value if 'system.disk.percent' in metrics else 0,
                ],
                "backgroundColor": [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 205, 86, 0.2)',
                ],
                "borderColor": [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 205, 86, 1)',
                ],
                "borderWidth": 1
            }]
        }

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the dashboard server"""
        uvicorn.run(self.app, host=host, port=port)


# Convenience function to create and run dashboard
def create_monitoring_dashboard(metrics_collector: MetricsCollector,
                               alert_manager: AlertManager,
                               pipeline_metrics: PipelineMetricsCollector) -> MonitoringDashboard:
    """Create a monitoring dashboard instance"""
    return MonitoringDashboard(metrics_collector, alert_manager, pipeline_metrics)
```

#### Step 2: Create Analytics System

```python
# analytics/pipeline_analytics.py
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


@dataclass
class AnalyticsReport:
    """Analytics report data structure"""
    report_type: str
    generated_at: datetime
    time_range: Dict[str, datetime]
    summary: Dict[str, Any]
    trends: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    data: Dict[str, Any]


class PipelineAnalytics:
    """Analytics system for crawling pipeline performance"""

    def __init__(self, database_connection, metrics_collector):
        self.db = database_connection
        self.metrics = metrics_collector

    async def generate_performance_report(self, days: int = 7) -> AnalyticsReport:
        """Generate comprehensive performance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Collect data
        crawling_stats = await self._get_crawling_stats(start_date, end_date)
        search_stats = await self._get_search_stats(start_date, end_date)
        system_stats = await self._get_system_stats(start_date, end_date)
        error_stats = await self._get_error_stats(start_date, end_date)

        # Analyze trends
        trends = await self._analyze_trends(crawling_stats, search_stats, days)

        # Generate insights
        insights = self._generate_insights(crawling_stats, search_stats, system_stats, error_stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(insights, trends)

        return AnalyticsReport(
            report_type="performance",
            generated_at=end_date,
            time_range={"start": start_date, "end": end_date},
            summary=self._create_summary(crawling_stats, search_stats, system_stats),
            trends=trends,
            insights=insights,
            recommendations=recommendations,
            data={
                "crawling": crawling_stats,
                "search": search_stats,
                "system": system_stats,
                "errors": error_stats
            }
        )

    async def _get_crawling_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get crawling performance statistics"""
        query = """
        SELECT
            DATE(crawled_at) as date,
            COUNT(*) as urls_crawled,
            COUNT(DISTINCT source_domain) as domains_crawled,
            AVG(word_count) as avg_word_count,
            COUNT(CASE WHEN language = 'en' THEN 1 END) as english_articles,
            COUNT(CASE WHEN language = 'fa' THEN 1 END) as persian_articles,
            AVG(EXTRACT(EPOCH FROM (crawled_at - LAG(crawled_at) OVER (ORDER BY crawled_at)))) as avg_time_between_crawls
        FROM crawled_articles
        WHERE crawled_at BETWEEN $1 AND $2
        GROUP BY DATE(crawled_at)
        ORDER BY date
        """

        rows = await self.db.fetch(query, start_date, end_date)

        # Calculate overall statistics
        if rows:
            df = pd.DataFrame([dict(row) for row in rows])
            overall = {
                'total_urls': int(df['urls_crawled'].sum()),
                'total_domains': int(df['domains_crawled'].max()),  # Max unique domains per day
                'avg_urls_per_day': float(df['urls_crawled'].mean()),
                'avg_word_count': float(df['avg_word_count'].mean()),
                'language_distribution': {
                    'english': int(df['english_articles'].sum()),
                    'persian': int(df['persian_articles'].sum())
                },
                'daily_stats': df.to_dict('records')
            }
        else:
            overall = {
                'total_urls': 0,
                'total_domains': 0,
                'avg_urls_per_day': 0,
                'avg_word_count': 0,
                'language_distribution': {'english': 0, 'persian': 0},
                'daily_stats': []
            }

        return overall

    async def _get_search_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get search performance statistics"""
        # Since we don't have direct search logs, we'll use metrics
        search_metrics = await self.metrics.get_metric_stats("pipeline.search.queries", time_range=int((end_date - start_date).total_seconds()))
        response_time_metrics = await self.metrics.get_metric_stats("pipeline.search.response_time", time_range=int((end_date - start_date).total_seconds()))

        return {
            'total_queries': search_metrics.last_value if search_metrics else 0,
            'avg_response_time': response_time_metrics.avg_value if response_time_metrics else 0,
            'queries_per_day': (search_metrics.last_value if search_metrics else 0) / max((end_date - start_date).days, 1)
        }

    async def _get_system_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get system resource usage statistics"""
        cpu_stats = await self.metrics.get_metric_stats("system.cpu.percent", time_range=int((end_date - start_date).total_seconds()))
        memory_stats = await self.metrics.get_metric_stats("system.memory.percent", time_range=int((end_date - start_date).total_seconds()))

        return {
            'avg_cpu_usage': cpu_stats.avg_value if cpu_stats else 0,
            'max_cpu_usage': cpu_stats.max_value if cpu_stats else 0,
            'avg_memory_usage': memory_stats.avg_value if memory_stats else 0,
            'max_memory_usage': memory_stats.max_value if memory_stats else 0
        }

    async def _get_error_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get error statistics"""
        # Get error metrics
        error_stats = await self.metrics.get_metric_stats("pipeline.errors.total", time_range=int((end_date - start_date).total_seconds()))

        # Get error breakdown (this would be enhanced with actual error logging)
        return {
            'total_errors': error_stats.last_value if error_stats else 0,
            'errors_per_day': (error_stats.last_value if error_stats else 0) / max((end_date - start_date).days, 1),
            'error_types': {}  # Would be populated from error logs
        }

    async def _analyze_trends(self, crawling_stats: Dict, search_stats: Dict, days: int) -> Dict[str, Any]:
        """Analyze performance trends"""
        trends = {}

        daily_stats = crawling_stats.get('daily_stats', [])

        if len(daily_stats) >= 2:
            # Calculate trends
            urls_trend = self._calculate_trend([d['urls_crawled'] for d in daily_stats])
            trends['urls_crawled'] = urls_trend

            # Calculate growth rates
            if len(daily_stats) >= 7:
                weekly_growth = (daily_stats[-1]['urls_crawled'] - daily_stats[0]['urls_crawled']) / daily_stats[0]['urls_crawled'] if daily_stats[0]['urls_crawled'] > 0 else 0
                trends['weekly_growth_rate'] = weekly_growth

        # Performance trends
        trends['search_load'] = 'increasing' if search_stats.get('queries_per_day', 0) > 100 else 'stable'

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _generate_insights(self, crawling_stats: Dict, search_stats: Dict,
                          system_stats: Dict, error_stats: Dict) -> List[str]:
        """Generate insights from the data"""
        insights = []

        # Crawling insights
        total_urls = crawling_stats.get('total_urls', 0)
        if total_urls > 10000:
            insights.append(f"High crawling volume: {total_urls} URLs processed")
        elif total_urls < 100:
            insights.append(f"Low crawling activity: only {total_urls} URLs processed")

        # Performance insights
        avg_response_time = search_stats.get('avg_response_time', 0)
        if avg_response_time > 5.0:
            insights.append(".2f")
        elif avg_response_time < 0.5:
            insights.append(".2f")

        # System insights
        avg_cpu = system_stats.get('avg_cpu_usage', 0)
        if avg_cpu > 80:
            insights.append(".1f")
        elif avg_cpu < 20:
            insights.append(".1f")

        # Error insights
        total_errors = error_stats.get('total_errors', 0)
        if total_errors > 100:
            insights.append(f"High error rate: {total_errors} errors detected")

        # Language insights
        lang_dist = crawling_stats.get('language_distribution', {})
        total_lang = sum(lang_dist.values())
        if total_lang > 0:
            persian_pct = lang_dist.get('persian', 0) / total_lang
            if persian_pct > 0.5:
                insights.append(".1%")

        return insights

    def _generate_recommendations(self, insights: List[str], trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on insights and trends"""
        recommendations = []

        # Based on insights
        for insight in insights:
            if 'High error rate' in insight:
                recommendations.append("Investigate and fix the root causes of high error rates")
                recommendations.append("Implement better error handling and retry mechanisms")
            elif 'Slow search response' in insight:
                recommendations.append("Optimize search indexes and query performance")
                recommendations.append("Consider implementing search result caching")
            elif 'High CPU usage' in insight:
                recommendations.append("Monitor and optimize CPU-intensive operations")
                recommendations.append("Consider scaling to multiple instances")
            elif 'Low crawling activity' in insight:
                recommendations.append("Review crawling configuration and target websites")
                recommendations.append("Check for blocking or rate limiting issues")

        # Based on trends
        urls_trend = trends.get('urls_crawled', 'stable')
        if urls_trend == 'decreasing':
            recommendations.append("Investigate declining crawling performance")
            recommendations.append("Check for website blocking or changes")

        weekly_growth = trends.get('weekly_growth_rate', 0)
        if weekly_growth < -0.2:
            recommendations.append("Address significant decline in crawling volume")

        return recommendations

    def _create_summary(self, crawling_stats: Dict, search_stats: Dict, system_stats: Dict) -> Dict[str, Any]:
        """Create summary statistics"""
        return {
            'crawling': {
                'total_urls': crawling_stats.get('total_urls', 0),
                'total_domains': crawling_stats.get('total_domains', 0),
                'avg_urls_per_day': crawling_stats.get('avg_urls_per_day', 0)
            },
            'search': {
                'total_queries': search_stats.get('total_queries', 0),
                'avg_response_time': search_stats.get('avg_response_time', 0)
            },
            'system': {
                'avg_cpu_usage': system_stats.get('avg_cpu_usage', 0),
                'avg_memory_usage': system_stats.get('avg_memory_usage', 0)
            }
        }

    async def export_report(self, report: AnalyticsReport, format: str = "json") -> str:
        """Export analytics report"""
        if format == "json":
            return json.dumps({
                'report_type': report.report_type,
                'generated_at': report.generated_at.isoformat(),
                'time_range': {
                    'start': report.time_range['start'].isoformat(),
                    'end': report.time_range['end'].isoformat()
                },
                'summary': report.summary,
                'trends': report.trends,
                'insights': report.insights,
                'recommendations': report.recommendations,
                'data': report.data
            }, indent=2, ensure_ascii=False)

        elif format == "markdown":
            return self._generate_markdown_report(report)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _generate_markdown_report(self, report: AnalyticsReport) -> str:
        """Generate markdown format report"""
        md = f"""# Crawling Pipeline Analytics Report

**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {report.time_range['start'].strftime('%Y-%m-%d')} to {report.time_range['end'].strftime('%Y-%m-%d')}

## Summary

### Crawling Performance
- Total URLs: {report.summary['crawling']['total_urls']:,}
- Total Domains: {report.summary['crawling']['total_domains']:,}
- Avg URLs/Day: {report.summary['crawling']['avg_urls_per_day']:.0f}

### Search Performance
- Total Queries: {report.summary['search']['total_queries']:,}
- Avg Response Time: {report.summary['search']['avg_response_time']:.2f}s

### System Resources
- Avg CPU Usage: {report.summary['system']['avg_cpu_usage']:.1f}%
- Avg Memory Usage: {report.summary['system']['avg_memory_usage']:.1f}%

## Key Insights

{chr(10).join(f"- {insight}" for insight in report.insights)}

## Recommendations

{chr(10).join(f"- {rec}" for rec in report.recommendations)}

## Trends

{chr(10).join(f"- **{key}:** {value}" for key, value in report.trends.items())}
"""

        return md
```

## Exercises

### Exercise 1: Metrics Collection Setup
1. Implement comprehensive metrics collection for your crawling pipeline
2. Set up system resource monitoring (CPU, memory, disk, network)
3. Create custom metrics for pipeline-specific operations
4. Implement metrics storage and historical data retention

### Exercise 2: Alert System Implementation
1. Create alert rules for critical pipeline conditions
2. Implement multiple notification channels (email, Slack, webhooks)
3. Set up alert escalation and auto-resolution
4. Test alert accuracy and reduce false positives

### Exercise 3: Dashboard Development
1. Build a web-based monitoring dashboard with real-time updates
2. Implement interactive charts and visualizations
3. Create drill-down capabilities for detailed metrics
4. Add export functionality for reports and data

### Exercise 4: Analytics and Reporting
1. Implement automated analytics report generation
2. Create performance trend analysis and forecasting
3. Develop insights engine for actionable recommendations
4. Build custom analytics queries and visualizations

## Next Steps
- Explore advanced monitoring techniques
- Learn about distributed system monitoring
- Study performance optimization strategies

## Resources
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/)
- [ELK Stack](https://www.elastic.co/what-is/elk-stack)
- [Application Performance Monitoring](https://www.datadoghq.com/apm/)
- [Time Series Databases](https://www.influxdata.com/time-series-database/)