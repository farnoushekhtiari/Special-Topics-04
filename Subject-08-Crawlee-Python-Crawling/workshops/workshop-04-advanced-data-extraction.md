# Workshop 04: Advanced Data Extraction Patterns and Strategies

## Overview
This workshop covers advanced data extraction patterns and strategies for complex websites. You'll learn to handle dynamic content, implement robust extraction pipelines, work with different data formats, and create scalable extraction systems that work across various website structures.

## Prerequisites
- Completed [Data Extraction Patterns Tutorial](../tutorials/04-data-extraction-patterns.md)
- Knowledge of basic HTML parsing and CSS selectors
- Understanding of different data formats (JSON, XML, etc.)

## Learning Objectives
By the end of this workshop, you will be able to:
- Implement advanced extraction patterns for complex websites
- Handle dynamic content and JavaScript-rendered pages
- Create robust extraction pipelines with error handling
- Work with various data formats and APIs
- Implement scalable and maintainable extraction systems

## Workshop Structure

### Part 1: Advanced Extraction Patterns

#### Step 1: Create Advanced Extractor Framework

```python
# extractors/advanced_extractor.py
import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union, Callable, Pattern
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse


@dataclass
class ExtractionRule:
    """Represents an extraction rule with pattern and processing"""
    name: str
    selector: str
    attribute: Optional[str] = None
    pattern: Optional[str] = None
    processor: Optional[Callable[[str], Any]] = None
    required: bool = False
    default_value: Any = None
    multiple: bool = False


@dataclass
class ExtractionResult:
    """Result of an extraction operation"""
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedExtractor(ABC):
    """Base class for advanced extractors"""

    def __init__(self):
        self.rules: List[ExtractionRule] = []
        self.compiled_patterns: Dict[str, Pattern] = {}

    def add_rule(self, rule: ExtractionRule):
        """Add an extraction rule"""
        self.rules.append(rule)

        # Pre-compile regex patterns
        if rule.pattern:
            self.compiled_patterns[rule.name] = re.compile(rule.pattern, re.IGNORECASE | re.MULTILINE)

    def extract(self, html_content: str, url: str = "") -> ExtractionResult:
        """Extract data using defined rules"""
        result = ExtractionResult()
        result.metadata['url'] = url
        result.metadata['extraction_timestamp'] = self._get_timestamp()

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for rule in self.rules:
                try:
                    extracted_value = self._extract_single_rule(soup, rule, url)

                    if extracted_value is not None:
                        result.data[rule.name] = extracted_value
                    elif rule.required:
                        result.errors.append(f"Required field '{rule.name}' not found")
                    elif rule.default_value is not None:
                        result.data[rule.name] = rule.default_value

                except Exception as e:
                    result.errors.append(f"Error extracting '{rule.name}': {str(e)}")

            # Validate extraction result
            self._validate_result(result)

        except Exception as e:
            result.errors.append(f"General extraction error: {str(e)}")

        return result

    def _extract_single_rule(self, soup: BeautifulSoup, rule: ExtractionRule, base_url: str) -> Any:
        """Extract data for a single rule"""
        # Find elements
        elements = soup.select(rule.selector)
        if not elements:
            return None

        # Extract values
        values = []
        for element in elements:
            value = self._extract_value_from_element(element, rule, base_url)
            if value is not None:
                values.append(value)

            if not rule.multiple:
                break

        # Process values
        if not values:
            return None

        if rule.multiple:
            return [self._process_value(v, rule) for v in values]
        else:
            return self._process_value(values[0], rule)

    def _extract_value_from_element(self, element: Tag, rule: ExtractionRule, base_url: str) -> Optional[str]:
        """Extract value from a single element"""
        if rule.attribute:
            # Extract from attribute
            if rule.attribute == 'text':
                return element.get_text(strip=True)
            elif rule.attribute == 'href':
                href = element.get('href')
                return urljoin(base_url, href) if href else None
            elif rule.attribute == 'src':
                src = element.get('src')
                return urljoin(base_url, src) if src else None
            else:
                return element.get(rule.attribute)
        else:
            # Extract text content
            return element.get_text(strip=True)

    def _process_value(self, value: str, rule: ExtractionRule) -> Any:
        """Process extracted value"""
        if not value:
            return value

        # Apply regex pattern if specified
        if rule.pattern and rule.name in self.compiled_patterns:
            match = self.compiled_patterns[rule.name].search(value)
            if match:
                if match.groups():
                    value = match.group(1)  # Return first capture group
                else:
                    value = match.group(0)

        # Apply custom processor
        if rule.processor:
            value = rule.processor(value)

        return value

    def _validate_result(self, result: ExtractionResult):
        """Validate extraction result"""
        # Check for required fields
        missing_required = [
            rule.name for rule in self.rules
            if rule.required and rule.name not in result.data
        ]

        if missing_required:
            result.errors.append(f"Missing required fields: {', '.join(missing_required)}")

        # Add validation metadata
        result.metadata['fields_extracted'] = len(result.data)
        result.metadata['errors_count'] = len(result.errors)
        result.metadata['success_rate'] = len(result.data) / len(self.rules) if self.rules else 0

    @abstractmethod
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        pass


class ProductExtractor(AdvancedExtractor):
    """Extractor for e-commerce product pages"""

    def __init__(self):
        super().__init__()
        self._setup_rules()

    def _setup_rules(self):
        """Setup extraction rules for products"""
        self.add_rule(ExtractionRule(
            name='title',
            selector='h1.product-title, .product-name, [class*="title"]',
            attribute='text',
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='price',
            selector='.price, .product-price, [class*="price"]',
            attribute='text',
            pattern=r'[\d,]+\.?\d*',
            processor=self._process_price,
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='description',
            selector='.product-description, .description, [class*="description"]',
            attribute='text'
        ))

        self.add_rule(ExtractionRule(
            name='images',
            selector='img.product-image, .product-gallery img, [class*="product"] img',
            attribute='src',
            multiple=True
        ))

        self.add_rule(ExtractionRule(
            name='sku',
            selector='.sku, .product-sku, [data-sku]',
            attribute='text',
            pattern=r'SKU:?\s*([A-Z0-9-]+)'
        ))

        self.add_rule(ExtractionRule(
            name='availability',
            selector='.availability, .stock-status, [class*="stock"]',
            attribute='text',
            processor=self._process_availability
        ))

        self.add_rule(ExtractionRule(
            name='categories',
            selector='.breadcrumb a, .categories a, [class*="category"] a',
            attribute='text',
            multiple=True
        ))

    def _process_price(self, price_str: str) -> Optional[float]:
        """Process price string to float"""
        try:
            # Remove currency symbols and commas
            clean_price = re.sub(r'[^\d.]', '', price_str)
            return float(clean_price)
        except (ValueError, TypeError):
            return None

    def _process_availability(self, availability_str: str) -> str:
        """Normalize availability status"""
        text = availability_str.lower().strip()

        if any(word in text for word in ['in stock', 'available', 'موجود']):
            return 'in_stock'
        elif any(word in text for word in ['out of stock', 'unavailable', 'ناموجود']):
            return 'out_of_stock'
        elif any(word in text for word in ['pre-order', 'پیش‌سفارش']):
            return 'pre_order'
        else:
            return 'unknown'

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


class NewsArticleExtractor(AdvancedExtractor):
    """Extractor for news articles"""

    def __init__(self):
        super().__init__()
        self._setup_rules()

    def _setup_rules(self):
        """Setup extraction rules for news articles"""
        self.add_rule(ExtractionRule(
            name='headline',
            selector='h1.article-title, .news-title, .headline',
            attribute='text',
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='subheadline',
            selector='.subheadline, .deck, .summary',
            attribute='text'
        ))

        self.add_rule(ExtractionRule(
            name='author',
            selector='.author, .byline, [rel="author"]',
            attribute='text',
            pattern=r'By:?\s*([^|,\n]+)'
        ))

        self.add_rule(ExtractionRule(
            name='publish_date',
            selector='time, .publish-date, .article-date',
            attribute='datetime',  # Try datetime attribute first
            default_value=None
        ))

        self.add_rule(ExtractionRule(
            name='content',
            selector='.article-content, .news-content, .post-content',
            attribute='text',
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='tags',
            selector='.tags a, .keywords a, .topics a',
            attribute='text',
            multiple=True
        ))

        self.add_rule(ExtractionRule(
            name='related_links',
            selector='.related-articles a, .related-links a',
            attribute='href',
            multiple=True
        ))

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


class JobPostingExtractor(AdvancedExtractor):
    """Extractor for job postings"""

    def __init__(self):
        super().__init__()
        self._setup_rules()

    def _setup_rules(self):
        """Setup extraction rules for job postings"""
        self.add_rule(ExtractionRule(
            name='job_title',
            selector='h1.job-title, .position-title, .job-header h1',
            attribute='text',
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='company',
            selector='.company-name, .employer, .job-company',
            attribute='text',
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='location',
            selector='.job-location, .location, .workplace',
            attribute='text'
        ))

        self.add_rule(ExtractionRule(
            name='salary',
            selector='.salary, .compensation, .pay',
            attribute='text',
            processor=self._process_salary
        ))

        self.add_rule(ExtractionRule(
            name='job_type',
            selector='.job-type, .employment-type, .work-type',
            attribute='text',
            processor=self._normalize_job_type
        ))

        self.add_rule(ExtractionRule(
            name='description',
            selector='.job-description, .job-details, .description',
            attribute='text',
            required=True
        ))

        self.add_rule(ExtractionRule(
            name='requirements',
            selector='.requirements, .qualifications, .job-requirements',
            attribute='text'
        ))

        self.add_rule(ExtractionRule(
            name='benefits',
            selector='.benefits, .perks, .job-benefits',
            attribute='text'
        ))

        self.add_rule(ExtractionRule(
            name='application_link',
            selector='.apply-button, .apply-link, a[href*="apply"]',
            attribute='href'
        ))

    def _process_salary(self, salary_str: str) -> Optional[Dict[str, Any]]:
        """Process salary information"""
        try:
            # Extract salary range
            range_pattern = r'\$?(\d+(?:,\d+)*)\s*(?:-\s*\$?(\d+(?:,\d+)*))?'
            match = re.search(range_pattern, salary_str)

            if match:
                min_salary = int(match.group(1).replace(',', ''))
                max_salary = int(match.group(2).replace(',', '')) if match.group(2) else None

                return {
                    'min': min_salary,
                    'max': max_salary,
                    'currency': 'USD' if '$' in salary_str else 'unknown'
                }
        except (ValueError, TypeError):
            pass

        return {'raw': salary_str}

    def _normalize_job_type(self, job_type_str: str) -> str:
        """Normalize job type"""
        text = job_type_str.lower().strip()

        if any(word in text for word in ['full.time', 'full-time', 'تمام وقت']):
            return 'full_time'
        elif any(word in text for word in ['part.time', 'part-time', 'پاره وقت']):
            return 'part_time'
        elif any(word in text for word in ['contract', 'freelance', 'قراردادی']):
            return 'contract'
        elif any(word in text for word in ['internship', 'intern', 'کارآموزی']):
            return 'internship'
        else:
            return text

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
```

#### Step 2: Create Dynamic Content Handler

```python
# extractors/dynamic_content_extractor.py
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from bs4 import BeautifulSoup
import re


class DynamicContentExtractor:
    """Extractor for JavaScript-rendered content"""

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start browser instance"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )

    async def stop(self):
        """Stop browser instance"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

    async def extract_page_content(self, url: str, wait_selector: Optional[str] = None,
                                  wait_time: int = 2000) -> Dict[str, Any]:
        """Extract content from a dynamic page"""
        if not self.context:
            raise RuntimeError("Browser not started. Use async context manager or call start() first.")

        page = await self.context.new_page()

        try:
            # Navigate to page
            await page.goto(url, timeout=self.timeout)

            # Wait for content to load
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=self.timeout)
            else:
                await page.wait_for_load_state('networkidle', timeout=self.timeout)

            # Additional wait for dynamic content
            await asyncio.sleep(wait_time / 1000)

            # Extract various content types
            content = {
                'url': url,
                'html': await page.content(),
                'text': await page.inner_text('body'),
                'title': await page.title()
            }

            # Extract structured data
            content.update(await self._extract_structured_data(page))

            # Extract dynamic elements
            content.update(await self._extract_dynamic_elements(page))

            return content

        finally:
            await page.close()

    async def _extract_structured_data(self, page: Page) -> Dict[str, Any]:
        """Extract JSON-LD and microdata"""
        structured_data = {}

        # Extract JSON-LD
        json_ld_scripts = await page.query_selector_all('script[type="application/ld+json"]')
        json_ld_data = []

        for script in json_ld_scripts:
            try:
                json_text = await script.inner_text()
                data = json.loads(json_text)
                json_ld_data.append(data)
            except (json.JSONDecodeError, Exception):
                continue

        if json_ld_data:
            structured_data['json_ld'] = json_ld_data

        # Extract meta tags
        meta_tags = await page.query_selector_all('meta')
        meta_data = {}

        for meta in meta_tags:
            name = await meta.get_attribute('name') or await meta.get_attribute('property')
            content = await meta.get_attribute('content')

            if name and content:
                meta_data[name] = content

        if meta_data:
            structured_data['meta_tags'] = meta_data

        return structured_data

    async def _extract_dynamic_elements(self, page: Page) -> Dict[str, Any]:
        """Extract dynamically loaded elements"""
        dynamic_data = {}

        # Common dynamic content selectors
        dynamic_selectors = {
            'infinite_scroll_items': '.item, .card, .post, [class*="item"]',
            'lazy_loaded_images': 'img[data-src], img[data-lazy-src]',
            'ajax_content': '[data-loaded], [data-ajax-loaded]',
            'spa_content': '[data-reactroot], [data-vue-instance]'
        }

        for name, selector in dynamic_selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    dynamic_data[name] = len(elements)
            except Exception:
                continue

        # Extract JavaScript variables
        js_vars = await page.evaluate("""
            () => {
                const variables = {};
                // Try to extract common SPA variables
                ['window.__INITIAL_STATE__', 'window.__NUXT__', 'window.__NEXT_DATA__'].forEach(varName => {
                    try {
                        const value = eval(varName);
                        if (value) variables[varName] = value;
                    } catch (e) {}
                });
                return variables;
            }
        """)

        if js_vars:
            dynamic_data['javascript_variables'] = js_vars

        return dynamic_data

    async def extract_api_calls(self, url: str, wait_time: int = 5000) -> List[Dict[str, Any]]:
        """Monitor and extract API calls made by the page"""
        if not self.context:
            raise RuntimeError("Browser not started.")

        page = await self.context.new_page()
        api_calls = []

        # Monitor network requests
        def handle_request(request):
            if any(api_indicator in request.url.lower() for api_indicator in ['/api/', '/graphql', '.json']):
                api_calls.append({
                    'url': request.url,
                    'method': request.method,
                    'headers': dict(request.headers),
                    'timestamp': asyncio.get_event_loop().time()
                })

        def handle_response(response):
            for call in api_calls:
                if call['url'] == response.url:
                    call['status'] = response.status
                    call['response_headers'] = dict(response.headers)
                    break

        page.on('request', handle_request)
        page.on('response', handle_response)

        try:
            await page.goto(url, timeout=self.timeout)
            await asyncio.sleep(wait_time / 1000)

            return api_calls

        finally:
            await page.close()

    async def take_screenshot(self, url: str, full_page: bool = True) -> bytes:
        """Take screenshot of the page"""
        if not self.context:
            raise RuntimeError("Browser not started.")

        page = await self.context.new_page()

        try:
            await page.goto(url, timeout=self.timeout)
            await page.wait_for_load_state('networkidle')

            screenshot = await page.screenshot(full_page=full_page)
            return screenshot

        finally:
            await page.close()


class SPAContentExtractor(DynamicContentExtractor):
    """Specialized extractor for Single Page Applications"""

    async def extract_spa_data(self, url: str) -> Dict[str, Any]:
        """Extract data from SPA applications"""
        content = await self.extract_page_content(url)

        spa_data = {
            'initial_state': None,
            'api_endpoints': [],
            'routes': []
        }

        # Extract common SPA data patterns
        html = content.get('html', '')

        # React/Next.js initial state
        next_data_match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if next_data_match:
            try:
                spa_data['initial_state'] = json.loads(next_data_match.group(1))
            except json.JSONDecodeError:
                pass

        # Vue.js initial state
        vue_state_match = re.search(r'window\.__NUXT__\s*=\s*({.*?});', html, re.DOTALL)
        if vue_state_match:
            try:
                spa_data['initial_state'] = json.loads(vue_state_match.group(1))
            except json.JSONDecodeError:
                pass

        # Extract API endpoints from JavaScript
        api_patterns = [
            r'["\'](/api/[^"\']+)["\']',
            r'["\'](https?://[^"\']+/api/[^"\']+)["\']',
            r'fetch\(["\']([^"\']+)["\']'
        ]

        for pattern in api_patterns:
            matches = re.findall(pattern, html)
            spa_data['api_endpoints'].extend(matches)

        # Remove duplicates
        spa_data['api_endpoints'] = list(set(spa_data['api_endpoints']))

        return {**content, **spa_data}
```

### Part 3: Data Format Handlers

#### Step 1: Create Format Handlers

```python
# extractors/format_handlers.py
import json
import xml.etree.ElementTree as ET
import csv
import io
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import yaml
import re


class DataFormatHandler(ABC):
    """Base class for data format handlers"""

    @abstractmethod
    def parse(self, content: str) -> Any:
        """Parse content into structured data"""
        pass

    @abstractmethod
    def can_handle(self, content: str) -> bool:
        """Check if this handler can process the content"""
        pass


class JSONHandler(DataFormatHandler):
    """Handler for JSON data"""

    def parse(self, content: str) -> Any:
        """Parse JSON content"""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def can_handle(self, content: str) -> bool:
        """Check if content is JSON"""
        content = content.strip()
        return (content.startswith('{') and content.endswith('}')) or \
               (content.startswith('[') and content.endswith(']'))


class XMLHandler(DataFormatHandler):
    """Handler for XML data"""

    def parse(self, content: str) -> Any:
        """Parse XML content"""
        try:
            root = ET.fromstring(content)

            # Convert XML to dictionary
            return self._xml_to_dict(root)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")

    def _xml_to_dict(self, element: ET.Element) -> Union[Dict, List, str]:
        """Convert XML element to dictionary"""
        # Handle text-only elements
        if not list(element):
            return element.text or ""

        # Handle elements with children
        result = {}

        for child in element:
            child_dict = self._xml_to_dict(child)

            if child.tag in result:
                # Multiple elements with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict

        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib

        return result

    def can_handle(self, content: str) -> bool:
        """Check if content is XML"""
        content = content.strip()
        return content.startswith('<') and content.endswith('>')


class CSVHandler(DataFormatHandler):
    """Handler for CSV data"""

    def __init__(self, delimiter: str = ',', has_header: bool = True):
        self.delimiter = delimiter
        self.has_header = has_header

    def parse(self, content: str) -> List[Dict[str, Any]]:
        """Parse CSV content"""
        try:
            csv_reader = csv.reader(io.StringIO(content), delimiter=self.delimiter)

            rows = list(csv_reader)
            if not rows:
                return []

            if self.has_header:
                headers = rows[0]
                data_rows = rows[1:]
            else:
                # Generate column names
                headers = [f'col_{i+1}' for i in range(len(rows[0]))]
                data_rows = rows

            return [
                {headers[i]: value for i, value in enumerate(row)}
                for row in data_rows
            ]

        except Exception as e:
            raise ValueError(f"Invalid CSV: {e}")

    def can_handle(self, content: str) -> bool:
        """Check if content looks like CSV"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check if lines have similar number of delimiters
        delimiter_counts = [line.count(self.delimiter) for line in lines[:5]]
        return len(set(delimiter_counts)) == 1 and delimiter_counts[0] > 0


class YAMLHandler(DataFormatHandler):
    """Handler for YAML data"""

    def parse(self, content: str) -> Any:
        """Parse YAML content"""
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML is required for YAML parsing")
        except Exception as e:
            raise ValueError(f"Invalid YAML: {e}")

    def can_handle(self, content: str) -> bool:
        """Check if content is YAML"""
        content = content.strip()
        # YAML files often start with --- or have key: value patterns
        return '---' in content[:100] or re.search(r'^[a-zA-Z_][a-zA-Z0-9_]*:\s', content, re.MULTILINE)


class HTMLTableHandler(DataFormatHandler):
    """Handler for HTML tables"""

    def parse(self, content: str) -> List[Dict[str, Any]]:
        """Parse HTML table content"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            tables = soup.find_all('table')

            if not tables:
                return []

            # Process first table
            table = tables[0]
            headers = self._extract_headers(table)
            rows = self._extract_rows(table, len(headers))

            return [
                {headers[i]: cell for i, cell in enumerate(row)}
                for row in rows
            ]

        except Exception as e:
            raise ValueError(f"Invalid HTML table: {e}")

    def _extract_headers(self, table) -> List[str]:
        """Extract table headers"""
        headers = []

        # Try thead first
        thead = table.find('thead')
        if thead:
            header_cells = thead.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]

        # Fallback to first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                header_cells = first_row.find_all(['th', 'td'])
                headers = [cell.get_text(strip=True) for cell in header_cells]

        # Generate default headers if none found
        if not headers:
            # Count columns from first data row
            rows = table.find_all('tr')
            if len(rows) > 1:
                cells = rows[1].find_all(['td', 'th'])
                headers = [f'col_{i+1}' for i in range(len(cells))]

        return headers

    def _extract_rows(self, table, num_columns: int) -> List[List[str]]:
        """Extract table rows"""
        rows = []
        tr_elements = table.find_all('tr')

        for tr in tr_elements:
            cells = tr.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]

            # Pad or truncate to match header count
            if len(row_data) < num_columns:
                row_data.extend([''] * (num_columns - len(row_data)))
            elif len(row_data) > num_columns:
                row_data = row_data[:num_columns]

            rows.append(row_data)

        # Skip header row if it was detected
        if rows and self._is_header_row(rows[0]):
            rows = rows[1:]

        return rows

    def _is_header_row(self, row: List[str]) -> bool:
        """Check if row looks like headers"""
        # Headers are typically shorter and more consistent
        return len(row) > 0 and all(len(cell) < 50 for cell in row)

    def can_handle(self, content: str) -> bool:
        """Check if content contains HTML tables"""
        soup = BeautifulSoup(content, 'html.parser')
        tables = soup.find_all('table')
        return len(tables) > 0


class DataFormatDetector:
    """Automatically detect and parse various data formats"""

    def __init__(self):
        self.handlers = [
            JSONHandler(),
            XMLHandler(),
            YAMLHandler(),
            CSVHandler(),
            HTMLTableHandler()
        ]

    def detect_and_parse(self, content: str) -> Any:
        """Detect format and parse content"""
        for handler in self.handlers:
            if handler.can_handle(content):
                try:
                    return handler.parse(content)
                except Exception:
                    continue

        # If no handler works, return raw content
        return content

    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats"""
        return [handler.__class__.__name__.replace('Handler', '').lower()
                for handler in self.handlers]


class APIResponseHandler:
    """Handler for API responses in various formats"""

    def __init__(self):
        self.format_detector = DataFormatDetector()

    def handle_response(self, response_text: str, content_type: Optional[str] = None) -> Any:
        """Handle API response based on content type"""
        if content_type:
            content_type = content_type.lower()

            if 'json' in content_type:
                return JSONHandler().parse(response_text)
            elif 'xml' in content_type:
                return XMLHandler().parse(response_text)
            elif 'yaml' in content_type or 'yml' in content_type:
                return YAMLHandler().parse(response_text)
            elif 'csv' in content_type:
                return CSVHandler().parse(response_text)
            elif 'html' in content_type:
                return HTMLTableHandler().parse(response_text)

        # Auto-detect if content-type not specified or not recognized
        return self.format_detector.detect_and_parse(response_text)
```

### Part 4: Scalable Extraction Pipeline

#### Step 1: Create Extraction Pipeline

```python
# pipelines/extraction_pipeline.py
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import json


@dataclass
class PipelineConfig:
    """Configuration for extraction pipeline"""
    max_concurrency: int = 5
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    output_directory: str = "extraction_output"
    save_intermediate: bool = True
    log_level: str = "INFO"


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExtractionPipeline:
    """Scalable extraction pipeline"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrency)
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ExtractionPipeline')

    async def run_pipeline(self, urls: List[str], extractor_func: Callable[[str], Awaitable[Any]]) -> PipelineResult:
        """Run extraction pipeline on multiple URLs"""
        start_time = time.time()

        result = PipelineResult()
        result.metadata['start_time'] = start_time
        result.metadata['total_urls'] = len(urls)

        self.logger.info(f"Starting pipeline with {len(urls)} URLs")

        # Create tasks with concurrency control
        tasks = []
        for url in urls:
            task = asyncio.create_task(self._process_single_url(url, extractor_func, result))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate timing
        end_time = time.time()
        result.timing = {
            'total_time': end_time - start_time,
            'avg_time_per_url': (end_time - start_time) / len(urls) if urls else 0,
            'urls_per_second': len(urls) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        }

        result.metadata['end_time'] = end_time

        self.logger.info(f"Pipeline completed: {result.successful}/{result.total_processed} successful")
        return result

    async def _process_single_url(self, url: str, extractor_func: Callable[[str], Awaitable[Any]],
                                  result: PipelineResult) -> None:
        """Process a single URL with retry logic"""
        async with self.semaphore:
            result.total_processed += 1

            for attempt in range(self.config.retry_attempts):
                try:
                    self.logger.debug(f"Processing {url} (attempt {attempt + 1})")

                    # Extract data
                    extracted_data = await extractor_func(url)

                    # Save intermediate results if configured
                    if self.config.save_intermediate:
                        await self._save_intermediate_result(url, extracted_data)

                    result.successful += 1
                    self.logger.debug(f"Successfully processed {url}")

                    # Rate limiting delay
                    await asyncio.sleep(self.config.rate_limit_delay)
                    return

                except Exception as e:
                    error_msg = f"Failed to process {url} (attempt {attempt + 1}): {str(e)}"
                    self.logger.warning(error_msg)

                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        result.failed += 1
                        result.errors.append(error_msg)

    async def _save_intermediate_result(self, url: str, data: Any):
        """Save intermediate extraction result"""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(exist_ok=True)

            # Create filename from URL
            filename = self._url_to_filename(url) + ".json"
            filepath = output_dir / filename

            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': url,
                    'timestamp': time.time(),
                    'data': data
                }, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save intermediate result for {url}: {e}")

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename"""
        import re
        from urllib.parse import urlparse

        # Extract domain and path
        parsed = urlparse(url)
        domain = parsed.netloc.replace('.', '_')
        path = parsed.path.strip('/').replace('/', '_')

        # Clean up special characters
        safe_name = re.sub(r'[^\w\-_]', '', f"{domain}_{path}")

        # Limit length
        return safe_name[:100] if safe_name else "unnamed"

    async def run_batch_extraction(self, url_batches: List[List[str]],
                                   extractor_func: Callable[[str], Awaitable[Any]]) -> List[PipelineResult]:
        """Run extraction on multiple batches sequentially"""
        results = []

        for i, batch in enumerate(url_batches):
            self.logger.info(f"Processing batch {i + 1}/{len(url_batches)} ({len(batch)} URLs)")
            batch_result = await self.run_pipeline(batch, extractor_func)
            results.append(batch_result)

            # Delay between batches
            if i < len(url_batches) - 1:
                await asyncio.sleep(1.0)

        return results

    def get_pipeline_stats(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """Aggregate statistics from multiple pipeline runs"""
        if not results:
            return {}

        total_processed = sum(r.total_processed for r in results)
        total_successful = sum(r.successful for r in results)
        total_failed = sum(r.failed for r in results)

        avg_time_per_url = sum(r.timing.get('avg_time_per_url', 0) for r in results) / len(results)
        total_time = sum(r.timing.get('total_time', 0) for r in results)

        return {
            'total_batches': len(results),
            'total_urls': total_processed,
            'successful': total_successful,
            'failed': total_failed,
            'success_rate': total_successful / total_processed if total_processed > 0 else 0,
            'avg_time_per_url': avg_time_per_url,
            'total_time': total_time,
            'urls_per_second': total_processed / total_time if total_time > 0 else 0
        }


class SpecializedPipeline(ExtractionPipeline):
    """Pipeline with specialized extractors"""

    def __init__(self, extractor_type: str = 'product', config: PipelineConfig = None):
        super().__init__(config)
        self.extractor_type = extractor_type
        self.extractor = self._get_extractor()

    def _get_extractor(self):
        """Get appropriate extractor based on type"""
        from extractors.advanced_extractor import ProductExtractor, NewsArticleExtractor, JobPostingExtractor

        extractors = {
            'product': ProductExtractor(),
            'news': NewsArticleExtractor(),
            'job': JobPostingExtractor()
        }

        return extractors.get(self.extractor_type, ProductExtractor())

    async def extract_single_url(self, url: str) -> Any:
        """Extract data from a single URL"""
        # This would need to be implemented to fetch HTML content
        # For now, return placeholder
        return {'url': url, 'extractor_type': self.extractor_type, 'status': 'placeholder'}

    async def run_typed_pipeline(self, urls: List[str]) -> PipelineResult:
        """Run pipeline with typed extractor"""
        return await self.run_pipeline(urls, self.extract_single_url)
```

## Exercises

### Exercise 1: Custom Extractor Development
1. Create a custom extractor for a specific website type (e.g., recipes, events, real estate)
2. Implement advanced CSS selectors and regex patterns
3. Add data validation and cleaning
4. Test with multiple pages from the same site

### Exercise 2: Dynamic Content Extraction
1. Set up Playwright for JavaScript-heavy websites
2. Extract content that loads after page interactions
3. Handle infinite scrolling and lazy loading
4. Monitor network requests for API endpoints

### Exercise 3: Data Format Integration
1. Create handlers for additional data formats (RSS, Atom, GraphQL)
2. Implement automatic format detection
3. Add data transformation and normalization
4. Handle compressed and encoded content

### Exercise 4: Scalable Pipeline Implementation
1. Build a distributed extraction pipeline
2. Implement queue-based processing
3. Add monitoring and alerting
4. Optimize for high-throughput scenarios

## Next Steps
- Complete [Workshop 05: Storage and Persistence](../workshops/workshop-05-storage-persistence.md)
- Learn about data processing and analysis
- Explore deployment and scaling strategies

## Resources
- [Advanced Web Scraping Techniques](https://example.com/advanced-scraping)
- [Data Extraction Patterns](https://example.com/extraction-patterns)
- [Playwright Documentation](https://playwright.dev/)
- [Beautiful Soup Advanced Usage](https://example.com/bs4-advanced)