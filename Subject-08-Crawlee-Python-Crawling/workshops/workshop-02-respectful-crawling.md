# Workshop 02: Respectful Crawling with Rate Limiting and Robots.txt

## Overview
This workshop implements respectful crawling practices using Crawlee Python SDK. You'll build a crawler that respects robots.txt files, implements proper rate limiting, and handles crawling delays to avoid being blocked by websites. This workshop focuses on ethical crawling behavior and anti-detection measures.

## Prerequisites
- Completed [Respectful Crawling Tutorial](../tutorials/02-respectful-crawling.md)
- Basic Crawlee knowledge from Workshop 01
- Python environment with Crawlee installed

## Learning Objectives
By the end of this workshop, you will be able to:
- Parse and respect robots.txt files
- Implement configurable rate limiting
- Add crawling delays and randomization
- Handle website blocking gracefully
- Monitor crawler behavior and adapt to restrictions

## Workshop Structure

### Part 1: Robots.txt Parser Implementation

#### Step 1: Create Robots.txt Parser

```python
# crawlers/respectful_crawler.py
import asyncio
import time
import random
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import httpx
from crawlee import ConcurrencyConfig
from crawlee.http_clients import HttpxHttpClient
from crawlee.storages import Dataset
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext


class RobotsTxtParser:
    """Parse and interpret robots.txt files"""

    def __init__(self, user_agent: str = 'RespectfulCrawler/1.0'):
        self.user_agent = user_agent
        self.parsers: Dict[str, RobotFileParser] = {}

    async def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        domain = self._get_domain(url)
        parser = await self._get_parser(domain)
        return parser.can_fetch(self.user_agent, url)

    async def get_crawl_delay(self, url: str) -> float:
        """Get crawl delay specified in robots.txt"""
        domain = self._get_domain(url)
        parser = await self._get_parser(domain)
        # RobotFileParser doesn't expose crawl delay, so we parse manually
        return await self._parse_crawl_delay(domain)

    async def _get_parser(self, domain: str) -> RobotFileParser:
        """Get or create parser for domain"""
        if domain not in self.parsers:
            parser = RobotFileParser()
            robots_url = f"https://{domain}/robots.txt"
            parser.set_url(robots_url)
            try:
                parser.read()
            except Exception as e:
                # If robots.txt doesn't exist or can't be read, assume allowed
                parser.allow_all = True
            self.parsers[domain] = parser
        return self.parsers[domain]

    async def _parse_crawl_delay(self, domain: str) -> float:
        """Parse crawl delay from robots.txt content"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://{domain}/robots.txt")
                if response.status_code == 200:
                    lines = response.text.split('\n')
                    current_user_agent = None
                    for line in lines:
                        line = line.strip().lower()
                        if line.startswith('user-agent:'):
                            current_user_agent = line.split(':', 1)[1].strip()
                        elif line.startswith('crawl-delay:') and current_user_agent in ['*', self.user_agent.lower()]:
                            try:
                                return float(line.split(':', 1)[1].strip())
                            except ValueError:
                                pass
        except Exception:
            pass
        return 1.0  # Default 1 second delay

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc


class RateLimiter:
    """Implement rate limiting for crawling"""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests: List[float] = []
        self.min_interval = 60.0 / requests_per_minute

    async def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()

        # Remove old requests outside the time window
        cutoff_time = current_time - 60.0
        self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until oldest request expires
            wait_time = 60.0 - (current_time - self.requests[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(time.time())

        # Add randomization to avoid patterns
        jitter = random.uniform(0.1, 1.0) * self.min_interval
        await asyncio.sleep(jitter)


class RespectfulCrawler:
    """Crawler that respects website policies and implements rate limiting"""

    def __init__(self, max_requests_per_minute: int = 30):
        self.robots_parser = RobotsTxtParser()
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.crawl_delays: Dict[str, float] = {}
        self.blocked_domains: Set[str] = set()

    async def crawl_website(self, start_urls: List[str], max_pages: int = 50):
        """Crawl websites respectfully"""

        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            url = context.request.url

            # Check robots.txt
            if not await self.robots_parser.can_fetch(url):
                print(f"Blocked by robots.txt: {url}")
                return

            # Check if domain is blocked
            domain = self.robots_parser._get_domain(url)
            if domain in self.blocked_domains:
                print(f"Domain blocked: {domain}")
                return

            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed()

                # Apply crawl delay
                if domain not in self.crawl_delays:
                    self.crawl_delays[domain] = await self.robots_parser.get_crawl_delay(url)
                await asyncio.sleep(self.crawl_delays[domain])

                # Extract data
                title = context.soup.title.string if context.soup.title else "No title"
                content = self._extract_content(context.soup)

                # Save data
                await context.push_data({
                    'url': url,
                    'title': title,
                    'content': content[:500],  # First 500 chars
                    'crawled_at': time.time()
                })

                print(f"Successfully crawled: {url}")

            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'blocked' in error_msg.lower():
                    # Rate limited or blocked
                    self.blocked_domains.add(domain)
                    print(f"Domain blocked due to rate limiting: {domain}")
                else:
                    print(f"Error crawling {url}: {error_msg}")

    def _extract_content(self, soup) -> str:
        """Extract main content from page"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try common content selectors
        content_selectors = [
            'article', '.content', '.post-content', '.entry-content',
            'main', '.main-content', '#content'
        ]

        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content.get_text(strip=True)

        # Fallback to body text
        return soup.body.get_text(strip=True) if soup.body else ""

    async def run(self, start_urls: List[str], max_pages: int = 50):
        """Run the respectful crawler"""
        crawler = BeautifulSoupCrawler(
            max_requests_per_crawl=max_pages,
            concurrency_config=ConcurrencyConfig(max_concurrency=1)  # Sequential for respectfulness
        )

        await crawler.run(start_urls, handler=self.crawl_website)


async def main():
    """Main function"""
    # Example usage
    start_urls = [
        'https://example.com',
        'https://httpbin.org/html'
    ]

    crawler = RespectfulCrawler(max_requests_per_minute=20)
    await crawler.run(start_urls, max_pages=10)

    # Export results
    dataset = await Dataset.open()
    data = await dataset.get_data()
    print(f"Crawled {len(data.items)} pages respectfully")


if __name__ == '__main__':
    asyncio.run(main())
```

### Part 2: Testing Respectful Behavior

#### Step 1: Create Test Script

```python
# tests/test_respectful_crawling.py
import asyncio
import time
from crawlers.respectful_crawler import RespectfulCrawler, RobotsTxtParser, RateLimiter


async def test_robots_parser():
    """Test robots.txt parsing"""
    parser = RobotsTxtParser()

    # Test with a real website
    test_urls = [
        'https://httpbin.org/html',
        'https://httpbin.org/json'
    ]

    for url in test_urls:
        can_fetch = await parser.can_fetch(url)
        delay = await parser.get_crawl_delay(url)
        print(f"{url}: can_fetch={can_fetch}, delay={delay}s")


async def test_rate_limiter():
    """Test rate limiting"""
    limiter = RateLimiter(requests_per_minute=10)

    start_time = time.time()
    for i in range(5):
        await limiter.wait_if_needed()
        print(f"Request {i+1} completed")

    elapsed = time.time() - start_time
    print(".2f")


async def test_full_crawler():
    """Test full respectful crawler"""
    crawler = RespectfulCrawler(max_requests_per_minute=5)

    start_urls = ['https://httpbin.org/html']
    await crawler.crawl_website(start_urls, max_pages=3)


if __name__ == '__main__':
    print("Testing Robots.txt Parser...")
    asyncio.run(test_robots_parser())

    print("\nTesting Rate Limiter...")
    asyncio.run(test_rate_limiter())

    print("\nTesting Full Crawler...")
    asyncio.run(test_full_crawler())
```

#### Step 2: Configuration File

```python
# config/respectful_config.py
from typing import Dict, Any

RESPECTFUL_CRAWLER_CONFIG = {
    'max_requests_per_minute': 30,
    'default_crawl_delay': 1.0,
    'user_agent': 'RespectfulCrawler/1.0 (+https://example.com/crawler)',
    'respect_robots_txt': True,
    'randomize_delays': True,
    'max_retries': 3,
    'retry_backoff_factor': 2.0,
    'blocked_domains_timeout': 3600,  # 1 hour
    'domains': {
        'example.com': {
            'max_requests_per_minute': 10,
            'crawl_delay': 2.0
        },
        'news-site.com': {
            'max_requests_per_minute': 5,
            'crawl_delay': 5.0
        }
    }
}

def get_domain_config(domain: str) -> Dict[str, Any]:
    """Get configuration for specific domain"""
    return RESPECTFUL_CRAWLER_CONFIG['domains'].get(domain, {})
```

### Part 3: Monitoring and Analytics

#### Step 1: Create Monitoring Module

```python
# monitoring/crawler_monitor.py
import time
import psutil
from typing import Dict, List, Any
from collections import defaultdict
import json


class CrawlerMonitor:
    """Monitor crawler performance and behavior"""

    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'requests_made': 0,
            'requests_blocked': 0,
            'requests_failed': 0,
            'domains_crawled': set(),
            'response_times': [],
            'errors_by_type': defaultdict(int),
            'rate_limit_hits': 0
        }

    def record_request(self, url: str, success: bool = True, response_time: float = 0.0, blocked: bool = False):
        """Record a request"""
        self.stats['requests_made'] += 1
        domain = url.split('/')[2] if '//' in url else 'unknown'
        self.stats['domains_crawled'].add(domain)

        if blocked:
            self.stats['requests_blocked'] += 1
        elif not success:
            self.stats['requests_failed'] += 1

        if response_time > 0:
            self.stats['response_times'].append(response_time)

    def record_error(self, error_type: str):
        """Record an error"""
        self.stats['errors_by_type'][error_type] += 1

    def record_rate_limit_hit(self):
        """Record rate limit hit"""
        self.stats['rate_limit_hits'] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get crawler statistics summary"""
        elapsed = time.time() - self.start_time

        return {
            'runtime_seconds': elapsed,
            'requests_per_second': self.stats['requests_made'] / elapsed if elapsed > 0 else 0,
            'total_requests': self.stats['requests_made'],
            'blocked_requests': self.stats['requests_blocked'],
            'failed_requests': self.stats['requests_failed'],
            'success_rate': (self.stats['requests_made'] - self.stats['requests_blocked'] - self.stats['requests_failed']) / self.stats['requests_made'] if self.stats['requests_made'] > 0 else 0,
            'domains_crawled': len(self.stats['domains_crawled']),
            'average_response_time': sum(self.stats['response_times']) / len(self.stats['response_times']) if self.stats['response_times'] else 0,
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'top_errors': dict(sorted(self.stats['errors_by_type'].items(), key=lambda x: x[1], reverse=True)[:5])
        }

    def save_report(self, filename: str = 'crawler_report.json'):
        """Save monitoring report to file"""
        report = {
            'timestamp': time.time(),
            'summary': self.get_summary(),
            'raw_stats': dict(self.stats)
        }

        # Convert sets to lists for JSON serialization
        report['raw_stats']['domains_crawled'] = list(report['raw_stats']['domains_crawled'])

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Crawler report saved to {filename}")


# Global monitor instance
crawler_monitor = CrawlerMonitor()
```

#### Step 2: Integrate Monitoring

```python
# crawlers/monitored_respectful_crawler.py
import asyncio
import time
from typing import List
from crawlers.respectful_crawler import RespectfulCrawler
from monitoring.crawler_monitor import crawler_monitor


class MonitoredRespectfulCrawler(RespectfulCrawler):
    """Respectful crawler with monitoring capabilities"""

    async def crawl_website(self, start_urls: List[str], max_pages: int = 50):
        """Monitored crawling with statistics"""

        async def handler(context):
            url = context.request.url
            start_time = time.time()

            try:
                # Check robots.txt
                if not await self.robots_parser.can_fetch(url):
                    crawler_monitor.record_request(url, blocked=True)
                    print(f"Blocked by robots.txt: {url}")
                    return

                # Check if domain is blocked
                domain = self.robots_parser._get_domain(url)
                if domain in self.blocked_domains:
                    crawler_monitor.record_request(url, blocked=True)
                    print(f"Domain blocked: {domain}")
                    return

                # Apply rate limiting and delays
                await self.rate_limiter.wait_if_needed()

                if domain not in self.crawl_delays:
                    self.crawl_delays[domain] = await self.robots_parser.get_crawl_delay(url)
                await asyncio.sleep(self.crawl_delays[domain])

                # Extract data
                title = context.soup.title.string if context.soup.title else "No title"
                content = self._extract_content(context.soup)

                # Save data
                await context.push_data({
                    'url': url,
                    'title': title,
                    'content': content[:500],
                    'crawled_at': time.time()
                })

                response_time = time.time() - start_time
                crawler_monitor.record_request(url, success=True, response_time=response_time)
                print(f"Successfully crawled: {url}")

            except Exception as e:
                response_time = time.time() - start_time
                crawler_monitor.record_request(url, success=False, response_time=response_time)

                error_msg = str(e)
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    crawler_monitor.record_rate_limit_hit()
                    self.blocked_domains.add(domain)
                    crawler_monitor.record_error('rate_limit')
                    print(f"Rate limited: {domain}")
                elif '403' in error_msg or 'blocked' in error_msg.lower():
                    self.blocked_domains.add(domain)
                    crawler_monitor.record_error('blocked')
                    print(f"Blocked: {domain}")
                else:
                    crawler_monitor.record_error('other')
                    print(f"Error crawling {url}: {error_msg}")


async def main():
    """Main function with monitoring"""
    start_urls = [
        'https://httpbin.org/html',
        'https://example.com'
    ]

    crawler = MonitoredRespectfulCrawler(max_requests_per_minute=15)
    await crawler.run(start_urls, max_pages=20)

    # Generate report
    crawler_monitor.save_report('respectful_crawler_report.json')

    # Print summary
    summary = crawler_monitor.get_summary()
    print("\n=== Crawler Summary ===")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
    print(f"Total requests: {summary['total_requests']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Requests/second: {summary['requests_per_second']:.2f}")
    print(f"Domains crawled: {summary['domains_crawled']}")
    print(f"Rate limit hits: {summary['rate_limit_hits']}")


if __name__ == '__main__':
    asyncio.run(main())
```

### Part 4: Advanced Configuration and Scaling

#### Step 1: Create Advanced Configuration

```python
# config/advanced_respectful_config.py
import os
from typing import Dict, Any, List
from urllib.parse import urlparse


class AdvancedCrawlerConfig:
    """Advanced configuration for respectful crawling"""

    def __init__(self):
        self.config = {
            'crawler': {
                'user_agent': 'AdvancedRespectfulCrawler/2.0 (+https://example.com/crawler)',
                'max_concurrency': 2,
                'max_requests_per_crawl': 100,
                'request_timeout': 30,
                'max_retries': 3
            },
            'rate_limiting': {
                'global_requests_per_minute': 60,
                'domain_specific_limits': {
                    'google.com': 10,
                    'twitter.com': 15,
                    'facebook.com': 20,
                    'news.ycombinator.com': 5
                },
                'adaptive_rate_limiting': True,
                'backoff_factor': 2.0
            },
            'delays': {
                'default_crawl_delay': 1.0,
                'randomize_delays': True,
                'min_delay': 0.5,
                'max_delay': 3.0,
                'domain_delays': {
                    'api.github.com': 2.0,
                    'httpbin.org': 0.5
                }
            },
            'robots_txt': {
                'respect_robots_txt': True,
                'cache_duration': 3600,  # 1 hour
                'ignore_missing_robots': False
            },
            'anti_detection': {
                'rotate_user_agents': True,
                'user_agents': [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                ],
                'randomize_headers': True,
                'session_persistence': False
            },
            'monitoring': {
                'enable_monitoring': True,
                'log_level': 'INFO',
                'metrics_interval': 60,
                'alert_on_blocked': True,
                'alert_on_rate_limit': True
            }
        }

        # Load from environment variables
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'CRAWLER_USER_AGENT': ('crawler', 'user_agent'),
            'MAX_REQUESTS_PER_MINUTE': ('rate_limiting', 'global_requests_per_minute'),
            'MAX_CONCURRENCY': ('crawler', 'max_concurrency'),
            'RESPECT_ROBOTS_TXT': ('robots_txt', 'respect_robots_txt')
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section, key = config_path
                # Convert string values to appropriate types
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)

                self.config[section][key] = value

    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific configuration"""
        domain_config = {}

        # Rate limiting
        if domain in self.config['rate_limiting']['domain_specific_limits']:
            domain_config['max_requests_per_minute'] = self.config['rate_limiting']['domain_specific_limits'][domain]

        # Delays
        if domain in self.config['delays']['domain_delays']:
            domain_config['crawl_delay'] = self.config['delays']['domain_delays'][domain]

        return domain_config

    def get_random_user_agent(self) -> str:
        """Get random user agent for anti-detection"""
        import random
        return random.choice(self.config['anti_detection']['user_agents'])

    def get_random_delay(self, base_delay: float) -> float:
        """Get randomized delay"""
        if not self.config['delays']['randomize_delays']:
            return base_delay

        import random
        min_delay = self.config['delays']['min_delay']
        max_delay = self.config['delays']['max_delay']

        # Random delay around base_delay
        factor = random.uniform(0.5, 2.0)
        delay = base_delay * factor

        # Clamp to min/max
        return max(min_delay, min(max_delay, delay))


# Global configuration instance
crawler_config = AdvancedCrawlerConfig()
```

## Exercises

### Exercise 1: Basic Respectful Crawler
1. Implement a crawler that respects robots.txt for a given website
2. Add rate limiting of 10 requests per minute
3. Test with at least 3 different websites
4. Log all blocked requests and reasons

### Exercise 2: Rate Limiting Strategies
1. Implement different rate limiting strategies (fixed, adaptive, domain-specific)
2. Compare performance and blocking rates
3. Implement exponential backoff for rate-limited requests
4. Test with high-traffic websites

### Exercise 3: Anti-Detection Measures
1. Implement user agent rotation
2. Add request header randomization
3. Implement session management
4. Test detection avoidance on various websites

### Exercise 4: Monitoring Dashboard
1. Create a simple web dashboard to monitor crawler statistics
2. Display real-time metrics (requests/minute, success rate, blocked domains)
3. Add alerts for rate limiting and blocks
4. Implement historical data tracking

## Next Steps
- Complete [Workshop 03: Persian Content Extraction](../workshops/workshop-03-persian-content-extraction.md)
- Learn about advanced data extraction patterns
- Explore storage and persistence options

## Resources
- [Crawlee Python Documentation](https://crawlee.dev/python/)
- [Robots.txt Specification](https://www.robotstxt.org/)
- [Web Crawling Ethics Guidelines](https://example.com/ethics)
- [Rate Limiting Best Practices](https://example.com/rate-limiting)