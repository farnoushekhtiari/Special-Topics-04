# Workshop 03: Persian Content Extraction and Processing

## Overview
This workshop focuses on building a specialized crawler for Persian (Farsi) content extraction. You'll implement text processing techniques for Persian websites, handle right-to-left text, manage character encoding issues, and create robust extraction pipelines for Persian news articles, blogs, and content sites.

## Prerequisites
- Completed [Persian Content Extraction Tutorial](../tutorials/03-persian-content-extraction.md)
- Knowledge of basic web scraping concepts
- Python environment with required libraries

## Learning Objectives
By the end of this workshop, you will be able to:
- Handle Persian text encoding and character normalization
- Extract content from Persian websites with RTL layout
- Implement Persian-specific text processing
- Create robust parsers for Persian news sites
- Handle mixed encoding scenarios

## Workshop Structure

### Part 1: Persian Text Processing Foundation

#### Step 1: Create Persian Text Utilities

```python
# utils/persian_text_utils.py
import re
import unicodedata
from typing import Dict, List, Set, Optional, Tuple
import html


class PersianTextProcessor:
    """Process and normalize Persian text"""

    # Persian characters and their variations
    PERSIAN_CHARS = {
        'ا': ['ا', 'أ', 'إ', 'آ'],
        'ه': ['ه', 'ة', 'ۀ'],
        'ی': ['ی', 'ي', 'ئ', 'ى'],
        'ک': ['ک', 'ك'],
        'گ': ['گ', 'گ'],
        'پ': ['پ', 'پ'],
        'ژ': ['ژ', 'ژ'],
        'چ': ['چ', 'چ']
    }

    # Persian numerals to Arabic numerals mapping
    PERSIAN_TO_ARABIC_NUMERALS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')

    def __init__(self):
        self.normalization_patterns = self._build_normalization_patterns()

    def _build_normalization_patterns(self) -> List[Tuple[str, str]]:
        """Build regex patterns for text normalization"""
        patterns = []

        # Persian character normalization
        for standard, variants in self.PERSIAN_CHARS.items():
            for variant in variants[1:]:  # Skip the first one (standard)
                patterns.append((re.escape(variant), standard))

        return patterns

    def normalize_persian_text(self, text: str) -> str:
        """Normalize Persian text characters"""
        if not text:
            return text

        # Apply character normalizations
        for pattern, replacement in self.normalization_patterns:
            text = re.sub(pattern, replacement, text)

        # Convert Persian numerals to Arabic numerals
        text = text.translate(self.PERSIAN_TO_ARABIC_NUMERALS)

        # Normalize Unicode (NFKC normalization)
        text = unicodedata.normalize('NFKC', text)

        return text

    def detect_persian_content(self, text: str) -> bool:
        """Detect if text contains significant Persian content"""
        if not text:
            return False

        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        total_chars = len([c for c in text if c.isalpha()])
        persian_chars_count = len([c for c in text if c in persian_chars])

        # Consider it Persian if >30% of alphabetic characters are Persian
        return total_chars > 0 and (persian_chars_count / total_chars) > 0.3

    def extract_persian_sentences(self, text: str) -> List[str]:
        """Extract Persian sentences from text"""
        # Persian sentence endings
        sentence_endings = r'[.!؟]\s+'

        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def clean_html_entities(self, text: str) -> str:
        """Clean HTML entities and decode Persian characters"""
        # Decode HTML entities
        text = html.unescape(text)

        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        return text

    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove whitespace around Persian punctuation
        persian_punct = '،؛؟«»'
        for punct in persian_punct:
            text = text.replace(f' {punct}', punct)
            text = text.replace(f'{punct} ', punct)

        return text.strip()


class PersianEncodingDetector:
    """Detect and handle Persian text encodings"""

    COMMON_ENCODINGS = [
        'utf-8',
        'windows-1256',  # Arabic/Windows Persian
        'iso-8859-6',    # ISO Arabic
        'cp1256',        # Windows Arabic
        'utf-16',
        'utf-32'
    ]

    def detect_encoding(self, content: bytes) -> Optional[str]:
        """Detect the encoding of Persian text content"""
        for encoding in self.COMMON_ENCODINGS:
            try:
                decoded = content.decode(encoding)
                # Check if decoding looks reasonable
                if self._is_reasonable_persian_text(decoded):
                    return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        return None

    def _is_reasonable_persian_text(self, text: str) -> bool:
        """Check if decoded text looks like reasonable Persian"""
        # Should contain some Persian characters
        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        has_persian = any(c in persian_chars for c in text)

        # Should not have too many replacement characters
        replacement_ratio = text.count('�') / len(text) if text else 1

        # Should be mostly printable characters
        printable_ratio = len([c for c in text if c.isprintable()]) / len(text) if text else 0

        return has_persian and replacement_ratio < 0.1 and printable_ratio > 0.8

    def decode_with_fallback(self, content: bytes, detected_encoding: Optional[str] = None) -> str:
        """Decode content with fallback handling"""
        if detected_encoding:
            try:
                return content.decode(detected_encoding)
            except UnicodeDecodeError:
                pass

        # Try UTF-8 first
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            pass

        # Try other encodings
        for encoding in self.COMMON_ENCODINGS[1:]:  # Skip UTF-8 as we tried it
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue

        # Last resort: use errors='replace'
        return content.decode('utf-8', errors='replace')
```

#### Step 2: Create Persian Content Extractor

```python
# extractors/persian_content_extractor.py
import re
from typing import Dict, List, Optional, Any, Tuple
from bs4 import BeautifulSoup, Tag
from utils.persian_text_utils import PersianTextProcessor, PersianEncodingDetector


class PersianContentExtractor:
    """Extract content from Persian websites"""

    def __init__(self):
        self.text_processor = PersianTextProcessor()
        self.encoding_detector = PersianEncodingDetector()

        # Persian website selectors (RTL-aware)
        self.content_selectors = [
            'article',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.news-content',
            '.blog-post',
            '[dir="rtl"]',
            '.rtl-content',
            '#content',
            'main'
        ]

        # Persian title selectors
        self.title_selectors = [
            'h1',
            '.title',
            '.headline',
            '.post-title',
            '.news-title',
            '.article-title',
            '[class*="title"]'
        ]

    def extract_from_html(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """Extract Persian content from HTML"""
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract basic metadata
            title = self._extract_title(soup)
            content = self._extract_main_content(soup)
            author = self._extract_author(soup)
            publish_date = self._extract_date(soup)
            tags = self._extract_tags(soup)

            # Process Persian text
            processed_content = self._process_persian_content(content)
            processed_title = self._process_persian_content(title)

            return {
                'url': url,
                'title': processed_title,
                'content': processed_content,
                'author': author,
                'publish_date': publish_date,
                'tags': tags,
                'language': 'fa' if self.text_processor.detect_persian_content(content) else 'unknown',
                'word_count': len(processed_content.split()),
                'sentence_count': len(self.text_processor.extract_persian_sentences(processed_content))
            }

        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'title': '',
                'content': '',
                'language': 'unknown'
            }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from Persian page"""
        for selector in self.title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 10:  # Reasonable title length
                    return title

        # Fallback to page title
        if soup.title:
            return soup.title.get_text(strip=True)

        return ""

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from Persian page"""
        # Remove unwanted elements
        self._clean_soup(soup)

        # Try content selectors
        for selector in self.content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = self._extract_text_from_element(content_element)
                if self._is_valid_content(content):
                    return content

        # Fallback: extract from body
        if soup.body:
            return self._extract_text_from_element(soup.body)

        return ""

    def _clean_soup(self, soup: BeautifulSoup):
        """Clean soup by removing unwanted elements"""
        unwanted_selectors = [
            'script', 'style', 'nav', 'header', 'footer',
            '.sidebar', '.advertisement', '.ads', '.comments',
            '.social-share', '.related-posts', '.navigation'
        ]

        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()

    def _extract_text_from_element(self, element: Tag) -> str:
        """Extract clean text from HTML element"""
        # Get text content
        text = element.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return self.text_processor.clean_html_entities(text)

    def _is_valid_content(self, content: str) -> bool:
        """Check if extracted content is valid"""
        if not content or len(content) < 50:
            return False

        # Should have some Persian content
        if not self.text_processor.detect_persian_content(content):
            return False

        # Should not be mostly navigation/menu text
        words = content.split()
        if len(words) < 20:
            return False

        # Check for reasonable sentence structure
        sentences = self.text_processor.extract_persian_sentences(content)
        if len(sentences) < 2:
            return False

        return True

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information"""
        author_selectors = [
            '.author', '.byline', '.writer', '.author-name',
            '[class*="author"]', '[rel="author"]'
        ]

        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text(strip=True)
                if author and len(author) > 2:
                    return self._process_persian_content(author)

        return None

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date"""
        date_selectors = [
            'time', '.date', '.published', '.post-date',
            '[datetime]', '[class*="date"]'
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                if element.get('datetime'):
                    return element['datetime']

                # Try text content
                date_text = element.get_text(strip=True)
                if date_text:
                    return date_text

        return None

    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract tags/keywords"""
        tags = []

        # Look for tag elements
        tag_elements = soup.select('.tag, .keyword, .category, [class*="tag"]')
        for element in tag_elements:
            tag_text = element.get_text(strip=True)
            if tag_text:
                tags.append(self._process_persian_content(tag_text))

        return tags[:10]  # Limit to 10 tags

    def _process_persian_content(self, text: str) -> str:
        """Process Persian text content"""
        if not text:
            return text

        # Clean HTML entities
        text = self.text_processor.clean_html_entities(text)

        # Normalize Persian characters
        text = self.text_processor.normalize_persian_text(text)

        # Clean whitespace
        text = self.text_processor.remove_extra_whitespace(text)

        return text


class PersianNewsExtractor(PersianContentExtractor):
    """Specialized extractor for Persian news websites"""

    def __init__(self):
        super().__init__()

        # News-specific selectors
        self.news_selectors = [
            '.news-article',
            '.news-content',
            '.story-content',
            '.article-body',
            '.news-body'
        ]

        self.lead_selectors = [
            '.lead', '.summary', '.abstract', '.excerpt',
            '.news-lead', '.article-lead'
        ]

    def extract_news_article(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """Extract Persian news article with specialized handling"""
        base_data = self.extract_from_html(html_content, url)

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract lead/summary
            lead = self._extract_lead(soup)
            if lead:
                base_data['lead'] = self._process_persian_content(lead)

            # Extract news-specific metadata
            category = self._extract_category(soup)
            if category:
                base_data['category'] = category

            source = self._extract_source(soup)
            if source:
                base_data['source'] = source

            # Mark as news content
            base_data['content_type'] = 'news'

        except Exception as e:
            base_data['extraction_error'] = str(e)

        return base_data

    def _extract_lead(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract news lead/summary"""
        for selector in self.lead_selectors:
            element = soup.select_one(selector)
            if element:
                lead = element.get_text(strip=True)
                if lead and len(lead) > 20:
                    return lead
        return None

    def _extract_category(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract news category"""
        category_selectors = [
            '.category', '.section', '.topic',
            '.news-category', '.article-category'
        ]

        for selector in category_selectors:
            element = soup.select_one(selector)
            if element:
                category = element.get_text(strip=True)
                if category:
                    return self._process_persian_content(category)
        return None

    def _extract_source(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract news source"""
        source_selectors = [
            '.source', '.news-source', '.publisher',
            '.article-source'
        ]

        for selector in source_selectors:
            element = soup.select_one(selector)
            if element:
                source = element.get_text(strip=True)
                if source:
                    return self._process_persian_content(source)
        return None
```

### Part 2: Persian Crawler Implementation

#### Step 1: Create Persian Content Crawler

```python
# crawlers/persian_content_crawler.py
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from crawlee import BeautifulSoupCrawler, BeautifulSoupCrawlingContext
from extractors.persian_content_extractor import PersianContentExtractor, PersianNewsExtractor
from utils.persian_text_utils import PersianTextProcessor


class PersianContentCrawler:
    """Crawler specialized for Persian content"""

    def __init__(self, max_pages: int = 50):
        self.max_pages = max_pages
        self.extractor = PersianContentExtractor()
        self.news_extractor = PersianNewsExtractor()
        self.text_processor = PersianTextProcessor()

        # Persian news websites
        self.persian_news_sites = {
            'isna.ir': 'Islamic Republic News Agency',
            'irna.ir': 'Iran News Agency',
            'mehrnews.com': 'Mehr News Agency',
            'farsnews.com': 'Fars News Agency',
            'tasnimnews.com': 'Tasnim News Agency',
            'yjc.ir': 'Young Journalists Club',
            'mizanonline.com': 'Mizan Online',
            'shana.ir': 'Shana News Agency'
        }

    async def crawl_persian_content(self, start_urls: List[str], content_type: str = 'general'):
        """Crawl Persian content from specified URLs"""

        results = []

        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            url = context.request.url

            try:
                # Check if content is Persian
                html_content = str(context.soup)
                if not self._is_persian_page(html_content):
                    print(f"Skipping non-Persian content: {url}")
                    return

                # Extract content based on type
                if content_type == 'news' or self._is_news_site(url):
                    extracted_data = self.news_extractor.extract_news_article(html_content, url)
                else:
                    extracted_data = self.extractor.extract_from_html(html_content, url)

                # Add crawling metadata
                extracted_data.update({
                    'crawled_at': context.request.loaded_at,
                    'crawl_depth': getattr(context.request, 'depth', 0),
                    'http_status': context.request.status_code
                })

                # Save data
                await context.push_data(extracted_data)
                results.append(extracted_data)

                print(f"Extracted Persian content: {extracted_data.get('title', 'No title')[:50]}...")

            except Exception as e:
                print(f"Error processing {url}: {e}")

        # Configure crawler for Persian content
        crawler = BeautifulSoupCrawler(
            max_requests_per_crawl=self.max_pages,
            request_handler_timeout=30
        )

        await crawler.run(start_urls, handler=handler)
        return results

    def _is_persian_page(self, html_content: str) -> bool:
        """Check if page contains Persian content"""
        # Quick check for Persian text
        return self.text_processor.detect_persian_content(html_content)

    def _is_news_site(self, url: str) -> bool:
        """Check if URL is from a Persian news site"""
        domain = urlparse(url).netloc.lower()
        return domain in self.persian_news_sites

    async def crawl_persian_news_sites(self, max_per_site: int = 10) -> List[Dict[str, Any]]:
        """Crawl multiple Persian news sites"""
        all_results = []

        for domain, site_name in self.persian_news_sites.items():
            print(f"Crawling {site_name} ({domain})...")

            try:
                # Get main page and find article links
                start_urls = [f"https://{domain}"]
                results = await self.crawl_persian_content(start_urls, content_type='news')

                # Limit results per site
                site_results = results[:max_per_site]
                all_results.extend(site_results)

                print(f"Extracted {len(site_results)} articles from {site_name}")

                # Add delay between sites
                await asyncio.sleep(2)

            except Exception as e:
                print(f"Error crawling {domain}: {e}")
                continue

        return all_results


class PersianBlogCrawler(PersianContentCrawler):
    """Crawler specialized for Persian blogs"""

    def __init__(self, max_pages: int = 30):
        super().__init__(max_pages)

        # Persian blog patterns
        self.blog_patterns = [
            r'/blog/',
            r'/post/',
            r'/article/',
            r'/news/',
            r'/\d{4}/\d{2}/\d{2}/',  # Date-based URLs
        ]

    async def crawl_persian_blog(self, blog_url: str) -> List[Dict[str, Any]]:
        """Crawl a Persian blog"""

        async def handler(context: BeautifulSoupCrawlingContext) -> None:
            url = context.request.url

            # Check if URL matches blog patterns
            if not self._is_blog_post(url):
                # Extract links to blog posts
                links = context.soup.select('a[href]')
                blog_links = []

                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(url, href)
                        if self._is_blog_post(full_url):
                            blog_links.append(full_url)

                # Add blog post links to crawler queue
                for link_url in blog_links[:5]:  # Limit to 5 per page
                    await context.add_requests([link_url])

                return

            # Process blog post
            html_content = str(context.soup)
            extracted_data = self.extractor.extract_from_html(html_content, url)

            await context.push_data(extracted_data)
            print(f"Extracted blog post: {extracted_data.get('title', 'No title')[:50]}...")

        crawler = BeautifulSoupCrawler(
            max_requests_per_crawl=self.max_pages,
            request_handler_timeout=30
        )

        await crawler.run([blog_url], handler=handler)

    def _is_blog_post(self, url: str) -> bool:
        """Check if URL is likely a blog post"""
        for pattern in self.blog_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False
```

### Part 3: Testing and Validation

#### Step 1: Create Test Suite

```python
# tests/test_persian_extraction.py
import asyncio
from extractors.persian_content_extractor import PersianContentExtractor, PersianNewsExtractor
from crawlers.persian_content_crawler import PersianContentCrawler


async def test_persian_text_processing():
    """Test Persian text processing utilities"""
    from utils.persian_text_utils import PersianTextProcessor

    processor = PersianTextProcessor()

    # Test Persian text normalization
    test_cases = [
        ("كتاب", "کتاب"),  # Kaf normalization
        ("يک", "یک"),      # Yeh normalization
        ("هٔ", "ه"),       # Heh normalization
        ("۱۲۳", "123"),    # Numeral conversion
    ]

    print("Testing Persian text normalization:")
    for input_text, expected in test_cases:
        result = processor.normalize_persian_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_text}' -> '{result}' (expected: '{expected}')")

    # Test Persian detection
    persian_text = "این یک متن فارسی است که شامل حروف فارسی می‌شود."
    english_text = "This is an English text."

    print("
Testing Persian detection:")
    print(f"  Persian text: {processor.detect_persian_content(persian_text)}")
    print(f"  English text: {processor.detect_persian_content(english_text)}")


async def test_content_extraction():
    """Test Persian content extraction"""
    extractor = PersianContentExtractor()

    # Test with sample HTML (simplified)
    sample_html = """
    <html dir="rtl" lang="fa">
    <head><title>عنوان مقاله فارسی</title></head>
    <body>
        <article class="content">
            <h1>عنوان مقاله فارسی</h1>
            <div class="author">نویسنده: علی محمدی</div>
            <div class="post-content">
                این یک مقاله نمونه فارسی است. متن مقاله شامل محتوای فارسی
                و ساختار مناسب برای استخراج است.
            </div>
        </article>
    </body>
    </html>
    """

    result = extractor.extract_from_html(sample_html, "https://example.com/article")

    print("Testing content extraction:")
    print(f"  Title: {result.get('title', 'N/A')}")
    print(f"  Language: {result.get('language', 'N/A')}")
    print(f"  Word count: {result.get('word_count', 0)}")


async def test_crawler():
    """Test Persian content crawler"""
    crawler = PersianContentCrawler(max_pages=5)

    # Test URLs (using safe test sites)
    test_urls = [
        "https://httpbin.org/html"  # Not Persian, but safe for testing
    ]

    print("Testing crawler (may not find Persian content):")
    try:
        results = await crawler.crawl_persian_content(test_urls)
        print(f"  Found {len(results)} results")
        for result in results[:2]:  # Show first 2
            print(f"    - {result.get('title', 'No title')[:30]}...")
    except Exception as e:
        print(f"  Error: {e}")


async def test_encoding_detection():
    """Test encoding detection"""
    from utils.persian_text_utils import PersianEncodingDetector

    detector = PersianEncodingDetector()

    # Test with UTF-8 Persian text
    persian_utf8 = "متن فارسی نمونه".encode('utf-8')
    detected = detector.detect_encoding(persian_utf8)

    print("Testing encoding detection:")
    print(f"  UTF-8 Persian text: {detected}")

    # Test decoding
    decoded = detector.decode_with_fallback(persian_utf8)
    print(f"  Decoded text: {decoded}")


async def main():
    """Run all tests"""
    print("=== Persian Content Extraction Tests ===\n")

    await test_persian_text_processing()
    print()

    await test_content_extraction()
    print()

    await test_encoding_detection()
    print()

    await test_crawler()
    print()

    print("Tests completed!")


if __name__ == '__main__':
    asyncio.run(main())
```

#### Step 2: Create Integration Test

```python
# tests/integration_test.py
import asyncio
import json
from pathlib import Path
from crawlers.persian_content_crawler import PersianContentCrawler, PersianBlogCrawler


async def run_integration_test():
    """Run comprehensive integration test"""
    print("=== Persian Content Crawler Integration Test ===\n")

    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Test 1: Basic Persian content crawler
    print("1. Testing basic Persian content crawler...")
    crawler = PersianContentCrawler(max_pages=10)

    # Use test URLs that might have Persian content
    test_urls = [
        "https://httpbin.org/html"  # Safe test endpoint
    ]

    try:
        results = await crawler.crawl_persian_content(test_urls)
        print(f"   Found {len(results)} Persian content items")

        # Save results
        with open(output_dir / "basic_crawler_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Persian blog crawler (simulated)
    print("\n2. Testing Persian blog crawler structure...")
    blog_crawler = PersianBlogCrawler(max_pages=5)

    # Test URL pattern matching
    test_blog_urls = [
        "https://example.com/blog/2023/01/sample-post",
        "https://example.com/article/123",
        "https://example.com/page"
    ]

    blog_posts = []
    non_blog_posts = []

    for url in test_blog_urls:
        if blog_crawler._is_blog_post(url):
            blog_posts.append(url)
        else:
            non_blog_posts.append(url)

    print(f"   Blog posts: {blog_posts}")
    print(f"   Non-blog posts: {non_blog_posts}")

    # Test 3: Content extraction quality
    print("\n3. Testing content extraction quality...")

    sample_persian_html = """
    <!DOCTYPE html>
    <html dir="rtl" lang="fa">
    <head>
        <title>اخبار ایران - مهمترین اخبار روز</title>
        <meta charset="utf-8">
    </head>
    <body>
        <header>
            <h1>پورتال خبری ایران</h1>
        </header>

        <main>
            <article class="news-article">
                <h1 class="news-title">رئیس جمهور ایران به اروپا سفر کرد</h1>
                <div class="news-meta">
                    <span class="author">به گزارش خبرنگار سیاسی</span>
                    <time datetime="2024-01-15">۱۵ دی ۱۴۰۲</time>
                    <span class="category">سیاسی</span>
                </div>

                <div class="news-lead">
                    رئیس جمهور اسلامی ایران امروز در ادامه سفرهای خارجی به کشور اروپایی سفر کرد.
                </div>

                <div class="news-content">
                    <p>در این سفر که به دعوت رسمی رئیس جمهور فرانسه انجام شده،
                    موضوعات مختلفی از جمله روابط دوجانبه، مسائل منطقه‌ای و بین‌المللی
                    مورد بحث و بررسی قرار خواهد گرفت.</p>

                    <p>رئیس جمهور در بدو ورود به فرودگاه مورد استقبال رسمی قرار گرفت
                    و سپس به محل اقامت خود منتقل شد.</p>
                </div>

                <div class="tags">
                    <span class="tag">رئیس جمهور</span>
                    <span class="tag">سفر خارجی</span>
                    <span class="tag">اروپا</span>
                </div>
            </article>
        </main>
    </body>
    </html>
    """

    extractor = PersianContentExtractor()
    extracted = extractor.extract_from_html(sample_persian_html, "https://news.example.com/politics/123")

    print("   Extracted data quality check:")
    print(f"   - Title: {'✓' if extracted.get('title') else '✗'}")
    print(f"   - Content: {'✓' if len(extracted.get('content', '')) > 50 else '✗'}")
    print(f"   - Language: {extracted.get('language', 'unknown')}")
    print(f"   - Word count: {extracted.get('word_count', 0)}")

    # Save extraction result
    with open(output_dir / "extraction_sample.json", 'w', encoding='utf-8') as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print("
4. Test Summary:")
    print(f"   - Output files saved to: {output_dir.absolute()}")
    print("   - Check the generated JSON files for detailed results")

    print("\n=== Integration Test Completed ===")


if __name__ == '__main__':
    asyncio.run(run_integration_test())
```

## Exercises

### Exercise 1: Persian Text Normalization
1. Implement character normalization for all Persian characters
2. Add support for Persian numeral conversion
3. Test with various Persian text samples
4. Handle mixed Arabic/Persian content

### Exercise 2: Content Extraction
1. Create extractors for different Persian website types (news, blogs, forums)
2. Implement robust content detection algorithms
3. Handle different Persian website layouts
4. Add support for Persian metadata extraction

### Exercise 3: Encoding Handling
1. Implement automatic encoding detection for Persian websites
2. Handle mixed encoding scenarios
3. Add fallback mechanisms for problematic encodings
4. Test with real Persian websites

### Exercise 4: Persian News Aggregator
1. Build a news aggregator for Persian news sites
2. Implement categorization and tagging
3. Add duplicate detection for news articles
4. Create a simple web interface to display results

## Next Steps
- Complete [Workshop 04: Advanced Data Extraction Patterns](../workshops/workshop-04-advanced-data-extraction.md)
- Learn about storage options for Persian content
- Explore Persian NLP processing techniques

## Resources
- [Persian Language Processing Resources](https://example.com/persian-nlp)
- [RTL Text Handling Guide](https://example.com/rtl-text)
- [Persian Web Standards](https://example.com/persian-web-standards)
- [Character Encoding in Persian](https://example.com/persian-encoding)