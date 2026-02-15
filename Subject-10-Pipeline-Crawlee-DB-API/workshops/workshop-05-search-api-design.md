# Workshop 05: Search API Design and Duplicate Handling

## Overview
This workshop focuses on building a comprehensive search API for crawled data with advanced duplicate detection and handling mechanisms. You'll implement full-text search capabilities, fuzzy matching algorithms, and real-time duplicate prevention to ensure data quality and efficient search operations.

## Prerequisites
- Completed [Search API Design Tutorial](../tutorials/05-search-api-design.md)
- Completed [Duplicate Handling Tutorial](../tutorials/04-duplicate-handling.md)
- Knowledge of REST API design and search algorithms

## Learning Objectives
By the end of this workshop, you will be able to:
- Design and implement RESTful search APIs
- Implement advanced duplicate detection algorithms
- Create fuzzy matching and similarity scoring
- Build real-time duplicate prevention systems
- Optimize search performance with indexing strategies

## Workshop Structure

### Part 1: Search API Architecture

#### Step 1: Create Search API Framework

```python
# api/search_api.py
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import re
import json


class SearchType(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    RELEVANCE = "relevance"
    DATE_DESC = "date_desc"
    DATE_ASC = "date_asc"
    TITLE_ASC = "title_asc"
    TITLE_DESC = "title_desc"


class SearchFilter(BaseModel):
    """Search filter model"""
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    source_domain: Optional[str] = None
    language: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    word_count_min: Optional[int] = Field(None, ge=0)
    word_count_max: Optional[int] = Field(None, ge=0)


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=500)
    search_type: SearchType = SearchType.HYBRID
    filters: SearchFilter = SearchFilter()
    sort_by: SortOrder = SortOrder.RELEVANCE
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    include_highlights: bool = True
    fuzzy_threshold: float = Field(0.6, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    url: str
    title: str
    content: str
    author: Optional[str]
    published_at: Optional[datetime]
    crawled_at: datetime
    source_domain: str
    language: str
    word_count: int
    tags: List[str]
    score: float
    highlights: Optional[Dict[str, List[str]]] = None
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    total_results: int
    results: List[SearchResult]
    search_time: float
    suggestions: List[str] = []
    facets: Dict[str, Any] = {}
    pagination: Dict[str, Union[int, bool]]


class DuplicateDetectionRequest(BaseModel):
    """Duplicate detection request"""
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    threshold: float = Field(0.85, ge=0.0, le=1.0)


class DuplicateResult(BaseModel):
    """Duplicate detection result"""
    is_duplicate: bool
    confidence: float
    duplicate_of: Optional[str] = None
    matches: List[Dict[str, Any]] = []


class SearchAPI:
    """Main search API class"""

    def __init__(self, database_connection, duplicate_detector):
        self.db = database_connection
        self.duplicate_detector = duplicate_detector
        self.app = FastAPI(title="Crawler Search API", version="1.0.0")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/")
        async def root():
            return {"message": "Crawler Search API", "version": "1.0.0"}

        @self.app.post("/search", response_model=SearchResponse)
        async def search(request: SearchRequest):
            """Main search endpoint"""
            start_time = datetime.now()

            try:
                # Perform search
                results_data = await self._perform_search(request)

                # Add search time
                search_time = (datetime.now() - start_time).total_seconds()

                # Generate suggestions if no results
                suggestions = []
                if not results_data["results"] and len(request.query.split()) == 1:
                    suggestions = await self._generate_suggestions(request.query)

                # Get facets
                facets = await self._get_search_facets(request.query, request.filters)

                # Build pagination info
                pagination = {
                    "offset": request.offset,
                    "limit": request.limit,
                    "has_more": len(results_data["results"]) == request.limit,
                    "total_pages": (results_data["total"] + request.limit - 1) // request.limit
                }

                return SearchResponse(
                    query=request.query,
                    total_results=results_data["total"],
                    results=results_data["results"],
                    search_time=search_time,
                    suggestions=suggestions,
                    facets=facets,
                    pagination=pagination
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.post("/detect-duplicate", response_model=DuplicateResult)
        async def detect_duplicate(request: DuplicateDetectionRequest):
            """Duplicate detection endpoint"""
            try:
                result = await self.duplicate_detector.detect_duplicate(
                    url=request.url,
                    title=request.title,
                    content=request.content,
                    threshold=request.threshold
                )

                return DuplicateResult(**result)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}")

        @self.app.get("/stats")
        async def get_stats():
            """Get search statistics"""
            try:
                stats = await self._get_search_stats()
                return stats
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Simple health check
                await self.db.fetchval("SELECT 1")
                return {"status": "healthy", "timestamp": datetime.now()}
            except Exception:
                raise HTTPException(status_code=503, detail="Service unhealthy")

    async def _perform_search(self, request: SearchRequest) -> Dict[str, Any]:
        """Perform the actual search"""
        # Build search query based on type
        if request.search_type == SearchType.EXACT:
            query_sql, params = self._build_exact_search_query(request)
        elif request.search_type == SearchType.FUZZY:
            query_sql, params = self._build_fuzzy_search_query(request)
        elif request.search_type == SearchType.SEMANTIC:
            query_sql, params = self._build_semantic_search_query(request)
        else:  # HYBRID
            query_sql, params = self._build_hybrid_search_query(request)

        # Execute search
        rows = await self.db.fetch(query_sql, *params)

        # Convert to SearchResult objects
        results = []
        for row in rows:
            result = SearchResult(
                id=str(row['id']),
                url=row['url'],
                title=row['title'],
                content=row['content'],
                author=row['author'],
                published_at=row['published_at'],
                crawled_at=row['crawled_at'],
                source_domain=row['source_domain'],
                language=row['language'],
                word_count=row['word_count'],
                tags=row['tags'] or [],
                score=float(row['score']) if 'score' in row else 1.0,
                highlights=row['highlights'] if 'highlights' in row else None,
                metadata=row['metadata'] or {}
            )
            results.append(result)

        # Get total count
        count_query = f"SELECT COUNT(*) FROM ({query_sql.replace('LIMIT', 'OFFSET').rsplit('LIMIT', 1)[0]}) AS subquery"
        total = await self.db.fetchval(count_query, *params)

        return {
            "results": results,
            "total": total
        }

    def _build_exact_search_query(self, request: SearchRequest) -> tuple[str, list]:
        """Build exact match search query"""
        base_query = """
        SELECT id, url, title, content, author, published_at, crawled_at,
               source_domain, language, word_count, tags, metadata,
               1.0 as score, NULL as highlights
        FROM crawled_articles
        WHERE content ILIKE $1
        """

        params = [f"%{request.query}%"]

        # Add filters
        filter_clauses, filter_params = self._build_filter_clauses(request.filters)
        if filter_clauses:
            base_query += " AND " + " AND ".join(filter_clauses)
            params.extend(filter_params)

        # Add sorting
        base_query += self._build_sort_clause(request.sort_by)

        # Add pagination
        base_query += f" LIMIT {request.limit} OFFSET {request.offset}"

        return base_query, params

    def _build_fuzzy_search_query(self, request: SearchRequest) -> tuple[str, list]:
        """Build fuzzy search query using trigram similarity"""
        base_query = """
        SELECT id, url, title, content, author, published_at, crawled_at,
               source_domain, language, word_count, tags, metadata,
               GREATEST(
                   similarity(title, $1),
                   similarity(content, $1)
               ) as score, NULL as highlights
        FROM crawled_articles
        WHERE (title % $1 OR content % $1)
           AND GREATEST(similarity(title, $1), similarity(content, $1)) > $2
        """

        params = [request.query, request.fuzzy_threshold]

        # Add filters
        filter_clauses, filter_params = self._build_filter_clauses(request.filters)
        if filter_clauses:
            base_query += " AND " + " AND ".join(filter_clauses)
            params.extend(filter_params)

        # Order by similarity score
        base_query += " ORDER BY score DESC"

        # Add pagination
        base_query += f" LIMIT {request.limit} OFFSET {request.offset}"

        return base_query, params

    def _build_semantic_search_query(self, request: SearchRequest) -> tuple[str, list]:
        """Build semantic search using full-text search"""
        base_query = """
        SELECT id, url, title, content, author, published_at, crawled_at,
               source_domain, language, word_count, tags, metadata,
               ts_rank(content_tsv, plainto_tsquery('english', $1)) as score,
               ts_headline('english', content, plainto_tsquery('english', $1)) as highlights
        FROM crawled_articles
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        """

        params = [request.query]

        # Add filters
        filter_clauses, filter_params = self._build_filter_clauses(request.filters)
        if filter_clauses:
            base_query += " AND " + " AND ".join(filter_clauses)
            params.extend(filter_params)

        # Order by relevance
        base_query += " ORDER BY score DESC"

        # Add pagination
        base_query += f" LIMIT {request.limit} OFFSET {request.offset}"

        return base_query, params

    def _build_hybrid_search_query(self, request: SearchRequest) -> tuple[str, list]:
        """Build hybrid search combining multiple techniques"""
        base_query = """
        SELECT id, url, title, content, author, published_at, crawled_at,
               source_domain, language, word_count, tags, metadata,
               (ts_rank(content_tsv, plainto_tsquery('english', $1)) +
                GREATEST(similarity(title, $1), similarity(content, $1))) / 2 as score,
               ts_headline('english', content, plainto_tsquery('english', $1)) as highlights
        FROM crawled_articles
        WHERE (content_tsv @@ plainto_tsquery('english', $1) OR
               title % $1 OR content % $1)
        """

        params = [request.query]

        # Add fuzzy threshold for similarity
        base_query += " AND GREATEST(similarity(title, $1), similarity(content, $1)) > $2"
        params.append(request.fuzzy_threshold)

        # Add filters
        filter_clauses, filter_params = self._build_filter_clauses(request.filters)
        if filter_clauses:
            base_query += " AND " + " AND ".join(filter_clauses)
            params.extend(filter_params)

        # Order by combined score
        base_query += " ORDER BY score DESC"

        # Add pagination
        base_query += f" LIMIT {request.limit} OFFSET {request.offset}"

        return base_query, params

    def _build_filter_clauses(self, filters: SearchFilter) -> tuple[List[str], List[Any]]:
        """Build WHERE clauses for filters"""
        clauses = []
        params = []

        if filters.date_from:
            clauses.append("published_at >= $" + str(len(params) + 1))
            params.append(filters.date_from)

        if filters.date_to:
            clauses.append("published_at <= $" + str(len(params) + 1))
            params.append(filters.date_to)

        if filters.source_domain:
            clauses.append("source_domain = $" + str(len(params) + 1))
            params.append(filters.source_domain)

        if filters.language:
            clauses.append("language = $" + str(len(params) + 1))
            params.append(filters.language)

        if filters.author:
            clauses.append("author ILIKE $" + str(len(params) + 1))
            params.append(f"%{filters.author}%")

        if filters.tags:
            # Use array overlap operator
            clauses.append("tags && $" + str(len(params) + 1))
            params.append(filters.tags)

        if filters.word_count_min is not None:
            clauses.append("word_count >= $" + str(len(params) + 1))
            params.append(filters.word_count_min)

        if filters.word_count_max is not None:
            clauses.append("word_count <= $" + str(len(params) + 1))
            params.append(filters.word_count_max)

        return clauses, params

    def _build_sort_clause(self, sort_by: SortOrder) -> str:
        """Build ORDER BY clause"""
        sort_mappings = {
            SortOrder.RELEVANCE: "score DESC",
            SortOrder.DATE_DESC: "published_at DESC NULLS LAST",
            SortOrder.DATE_ASC: "published_at ASC NULLS LAST",
            SortOrder.TITLE_ASC: "title ASC",
            SortOrder.TITLE_DESC: "title DESC"
        }

        return f" ORDER BY {sort_mappings[sort_by]}"

    async def _generate_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions"""
        # Find similar terms in the database
        suggestions_query = """
        SELECT word, count(*) as freq
        FROM (
            SELECT unnest(tsvector_to_array(content_tsv)) as word
            FROM crawled_articles
            WHERE content_tsv @@ plainto_tsquery('english', $1)
            LIMIT 1000
        ) AS words
        WHERE word LIKE $2 || '%'
          AND length(word) > 3
        GROUP BY word
        ORDER BY freq DESC, word
        LIMIT 5
        """

        pattern = query.split()[-1] if query.split() else query
        rows = await self.db.fetch(suggestions_query, query, pattern)

        return [row['word'] for row in rows]

    async def _get_search_facets(self, query: str, filters: SearchFilter) -> Dict[str, Any]:
        """Get search facets for filtering"""
        facets = {}

        # Language distribution
        lang_query = """
        SELECT language, COUNT(*) as count
        FROM crawled_articles
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        GROUP BY language
        ORDER BY count DESC
        LIMIT 10
        """
        lang_rows = await self.db.fetch(lang_query, query)
        facets['languages'] = {row['language']: row['count'] for row in lang_rows}

        # Domain distribution
        domain_query = """
        SELECT source_domain, COUNT(*) as count
        FROM crawled_articles
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        GROUP BY source_domain
        ORDER BY count DESC
        LIMIT 10
        """
        domain_rows = await self.db.fetch(domain_query, query)
        facets['domains'] = {row['source_domain']: row['count'] for row in domain_rows}

        # Date ranges
        date_query = """
        SELECT
            DATE_TRUNC('month', published_at) as month,
            COUNT(*) as count
        FROM crawled_articles
        WHERE content_tsv @@ plainto_tsquery('english', $1)
          AND published_at IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
        LIMIT 12
        """
        date_rows = await self.db.fetch(date_query, query)
        facets['date_ranges'] = {row['month'].isoformat(): row['count'] for row in date_rows}

        return facets

    async def _get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        stats_query = """
        SELECT
            COUNT(*) as total_articles,
            COUNT(DISTINCT source_domain) as unique_domains,
            COUNT(DISTINCT language) as unique_languages,
            AVG(word_count) as avg_word_count,
            MIN(published_at) as oldest_article,
            MAX(published_at) as newest_article
        FROM crawled_articles
        """

        stats = await self.db.fetchrow(stats_query)

        # Get popular search terms (mock data for now)
        popular_terms = [
            "python", "web scraping", "data science", "machine learning",
            "api", "database", "tutorial", "guide"
        ]

        return {
            'total_articles': stats['total_articles'],
            'unique_domains': stats['unique_domains'],
            'unique_languages': stats['unique_languages'],
            'avg_word_count': stats['avg_word_count'],
            'date_range': {
                'oldest': stats['oldest_article'].isoformat() if stats['oldest_article'] else None,
                'newest': stats['newest_article'].isoformat() if stats['newest_article'] else None
            },
            'popular_search_terms': popular_terms
        }
```

#### Step 2: Implement Duplicate Detection System

```python
# deduplication/detector.py
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import difflib
from collections import Counter


@dataclass
class DuplicateCandidate:
    """Candidate duplicate article"""
    id: str
    url: str
    title: str
    content_hash: str
    similarity_score: float
    match_type: str  # 'exact', 'near_exact', 'similar'


@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection"""
    is_duplicate: bool
    confidence: float
    duplicate_of: Optional[str] = None
    candidates: List[DuplicateCandidate] = None

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


class ContentNormalizer:
    """Normalize content for duplicate detection"""

    def __init__(self):
        # Patterns to remove for normalization
        self.noise_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # dates
            r'\d{1,2}:\d{2}',  # times
            r'https?://[^\s]+',  # URLs
            r'[^\w\s]',  # punctuation
            r'\s+',  # extra whitespace
        ]

        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would'
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove noise patterns
        for pattern in self.noise_patterns[:-1]:  # Don't remove whitespace yet
            text = re.sub(pattern, '', text)

        # Split into words
        words = text.split()

        # Remove stop words and short words
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]

        # Join back
        normalized = ' '.join(filtered_words)

        # Final whitespace cleanup
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        normalized = self.normalize_text(text)
        words = normalized.split()

        # Count word frequencies
        word_counts = Counter(words)

        # Get most common words
        keywords = [word for word, count in word_counts.most_common(max_keywords)]

        return keywords


class SimilarityCalculator:
    """Calculate similarity between texts"""

    def __init__(self):
        self.normalizer = ContentNormalizer()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        norm1 = self.normalizer.normalize_text(text1)
        norm2 = self.normalizer.normalize_text(text2)

        if not norm1 or not norm2:
            return 0.0

        # Calculate different similarity metrics
        exact_match = 1.0 if norm1 == norm2 else 0.0
        sequence_match = difflib.SequenceMatcher(None, norm1, norm2).ratio()

        # Keyword overlap
        keywords1 = set(self.normalizer.extract_keywords(text1, 20))
        keywords2 = set(self.normalizer.extract_keywords(text2, 20))
        keyword_overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2) if (keywords1 | keywords2) else 0.0

        # Weighted combination
        similarity = (exact_match * 0.4 + sequence_match * 0.4 + keyword_overlap * 0.2)

        return min(similarity, 1.0)  # Cap at 1.0

    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity specifically for titles"""
        if not title1 or not title2:
            return 0.0

        # Normalize titles
        norm1 = self.normalizer.normalize_text(title1)
        norm2 = self.normalizer.normalize_text(title2)

        # Exact match gets high score
        if norm1 == norm2:
            return 1.0

        # Sequence matching for titles
        return difflib.SequenceMatcher(None, norm1, norm2).ratio()


class DuplicateDetector:
    """Main duplicate detection system"""

    def __init__(self, database_connection):
        self.db = database_connection
        self.similarity_calculator = SimilarityCalculator()
        self.content_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = 1000

    async def detect_duplicate(self, url: Optional[str] = None,
                             title: Optional[str] = None,
                             content: Optional[str] = None,
                             threshold: float = 0.85) -> Dict[str, Any]:
        """Detect if content is duplicate"""

        if not any([url, title, content]):
            return {
                'is_duplicate': False,
                'confidence': 0.0,
                'duplicate_of': None,
                'matches': []
            }

        # Generate content hash for quick lookup
        content_hash = self._generate_content_hash(content or "")

        # Check exact hash match first
        existing = await self.db.fetchrow(
            "SELECT id, url, title FROM crawled_articles WHERE content_hash = $1",
            content_hash
        )

        if existing:
            return {
                'is_duplicate': True,
                'confidence': 1.0,
                'duplicate_of': str(existing['id']),
                'matches': [{
                    'id': str(existing['id']),
                    'url': existing['url'],
                    'title': existing['title'],
                    'similarity_score': 1.0,
                    'match_type': 'exact'
                }]
            }

        # Find similar content
        candidates = await self._find_similar_content(url, title, content, threshold)

        if not candidates:
            return {
                'is_duplicate': False,
                'confidence': 0.0,
                'duplicate_of': None,
                'matches': []
            }

        # Sort by similarity score
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        best_match = candidates[0]

        return {
            'is_duplicate': best_match.similarity_score >= threshold,
            'confidence': best_match.similarity_score,
            'duplicate_of': best_match.id if best_match.similarity_score >= threshold else None,
            'matches': [{
                'id': candidate.id,
                'url': candidate.url,
                'title': candidate.title,
                'similarity_score': candidate.similarity_score,
                'match_type': candidate.match_type
            } for candidate in candidates[:5]]  # Top 5 matches
        }

    async def _find_similar_content(self, url: Optional[str],
                                   title: Optional[str],
                                   content: Optional[str],
                                   threshold: float) -> List[DuplicateCandidate]:
        """Find similar content in database"""
        candidates = []

        # Strategy 1: Similar titles (fastest)
        if title:
            title_candidates = await self._find_by_similar_title(title, threshold)
            candidates.extend(title_candidates)

        # Strategy 2: URL pattern matching
        if url:
            url_candidates = await self._find_by_url_pattern(url)
            candidates.extend(url_candidates)

        # Strategy 3: Content similarity (most expensive, use as fallback)
        if content and len(candidates) < 3:
            content_candidates = await self._find_by_content_similarity(content, threshold)
            candidates.extend(content_candidates)

        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.id not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate.id)

        return unique_candidates

    async def _find_by_similar_title(self, title: str, threshold: float) -> List[DuplicateCandidate]:
        """Find articles with similar titles"""
        # Use trigram similarity for fuzzy title matching
        query = """
        SELECT id, url, title, content_hash
        FROM crawled_articles
        WHERE title % $1 AND similarity(title, $1) > $2
        ORDER BY similarity(title, $1) DESC
        LIMIT 10
        """

        rows = await self.db.fetch(query, title, threshold * 0.8)  # Lower threshold for titles

        candidates = []
        for row in rows:
            similarity = self.similarity_calculator.calculate_title_similarity(title, row['title'])
            if similarity >= threshold:
                candidates.append(DuplicateCandidate(
                    id=str(row['id']),
                    url=row['url'],
                    title=row['title'],
                    content_hash=row['content_hash'],
                    similarity_score=similarity,
                    match_type='near_exact' if similarity > 0.95 else 'similar'
                ))

        return candidates

    async def _find_by_url_pattern(self, url: str) -> List[DuplicateCandidate]:
        """Find articles with similar URL patterns"""
        # Extract domain and path pattern
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        path_parts = parsed.path.strip('/').split('/')[:3]  # First 3 path segments

        # Look for URLs from same domain with similar paths
        query = """
        SELECT id, url, title, content_hash
        FROM crawled_articles
        WHERE source_domain = $1
          AND url LIKE $2
        LIMIT 5
        """

        path_pattern = f"https://{domain}/{'/'.join(path_parts)}%"
        rows = await self.db.fetch(query, domain, path_pattern)

        candidates = []
        for row in rows:
            candidates.append(DuplicateCandidate(
                id=str(row['id']),
                url=row['url'],
                title=row['title'],
                content_hash=row['content_hash'],
                similarity_score=0.9,  # High confidence for URL pattern matches
                match_type='url_pattern'
            ))

        return candidates

    async def _find_by_content_similarity(self, content: str, threshold: float) -> List[DuplicateCandidate]:
        """Find articles with similar content using full-text search"""
        # Use full-text search to find candidates
        query = """
        SELECT id, url, title, content, content_hash
        FROM crawled_articles
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $1)) DESC
        LIMIT 20
        """

        # Extract search terms from content
        search_terms = self.similarity_calculator.normalizer.extract_keywords(content, 5)
        search_query = ' '.join(search_terms)

        rows = await self.db.fetch(query, search_query)

        candidates = []
        for row in rows:
            similarity = self.similarity_calculator.calculate_similarity(content, row['content'])
            if similarity >= threshold:
                candidates.append(DuplicateCandidate(
                    id=str(row['id']),
                    url=row['url'],
                    title=row['title'],
                    content_hash=row['content_hash'],
                    similarity_score=similarity,
                    match_type='content_similarity'
                ))

        return candidates

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        if not content:
            return ""

        # Normalize content first
        normalized = self.similarity_calculator.normalizer.normalize_text(content)

        # Generate hash
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    async def batch_detect_duplicates(self, articles: List[Dict[str, Any]],
                                    threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Detect duplicates in a batch of articles"""
        results = []

        # Process in smaller batches to avoid overwhelming the database
        batch_size = 10

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]

            # Process batch concurrently
            tasks = []
            for article in batch:
                task = self.detect_duplicate(
                    url=article.get('url'),
                    title=article.get('title'),
                    content=article.get('content'),
                    threshold=threshold
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    async def cleanup_duplicates(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up duplicate articles"""
        # Find all duplicates
        query = """
        SELECT content_hash, COUNT(*) as count, array_agg(id) as ids
        FROM crawled_articles
        GROUP BY content_hash
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        """

        duplicate_groups = await self.db.fetch(query)

        stats = {
            'total_duplicate_groups': len(duplicate_groups),
            'total_duplicate_articles': sum(row['count'] - 1 for row in duplicate_groups),
            'deleted_articles': 0,
            'groups_processed': 0
        }

        if dry_run:
            return {**stats, 'dry_run': True}

        # Delete duplicates (keep the first one in each group)
        for group in duplicate_groups:
            ids = group['ids'][1:]  # Skip the first ID (keep it)

            if ids:
                # Delete duplicates
                await self.db.execute(
                    "DELETE FROM crawled_articles WHERE id = ANY($1)",
                    ids
                )

                stats['deleted_articles'] += len(ids)

            stats['groups_processed'] += 1

        return {**stats, 'dry_run': False}
```

### Part 2: Search API Testing and Validation

#### Step 1: Create API Test Suite

```python
# tests/test_search_api.py
import asyncio
import pytest
from httpx import AsyncClient
from api.search_api import SearchAPI, SearchRequest, SearchFilter, SearchType
from deduplication.detector import DuplicateDetector


class TestSearchAPI:
    """Test suite for search API"""

    def setup_method(self):
        """Setup test environment"""
        # Mock database connection
        self.mock_db = MockDatabase()
        self.duplicate_detector = DuplicateDetector(self.mock_db)

        # Create API instance
        self.search_api = SearchAPI(self.mock_db, self.duplicate_detector)
        self.client = AsyncClient(app=self.search_api.app, base_url="http://testserver")

    async def test_basic_search(self):
        """Test basic search functionality"""
        # Add test data
        await self.mock_db.add_test_articles([
            {
                'id': '1',
                'title': 'Python Programming Guide',
                'content': 'Learn Python programming from basics to advanced',
                'url': 'https://example.com/python-guide',
                'author': 'John Doe',
                'published_at': '2024-01-15T10:00:00',
                'source_domain': 'example.com',
                'language': 'en',
                'word_count': 150,
                'tags': ['python', 'programming']
            },
            {
                'id': '2',
                'title': 'Web Scraping Tutorial',
                'content': 'Master web scraping with Python and BeautifulSoup',
                'url': 'https://example.com/scraping-tutorial',
                'author': 'Jane Smith',
                'published_at': '2024-01-20T14:30:00',
                'source_domain': 'example.com',
                'language': 'en',
                'word_count': 200,
                'tags': ['python', 'scraping']
            }
        ])

        # Test search request
        request = SearchRequest(query="python programming")
        response = await self.client.post("/search", json=request.dict())

        assert response.status_code == 200
        data = response.json()

        assert data['total_results'] >= 1
        assert len(data['results']) >= 1
        assert data['results'][0]['title'] == 'Python Programming Guide'

    async def test_filtered_search(self):
        """Test search with filters"""
        request = SearchRequest(
            query="python",
            filters=SearchFilter(
                author="John Doe",
                date_from="2024-01-01T00:00:00",
                tags=["python"]
            )
        )

        response = await self.client.post("/search", json=request.dict())

        assert response.status_code == 200
        data = response.json()

        # Should find John's article
        assert any(r['author'] == 'John Doe' for r in data['results'])

    async def test_fuzzy_search(self):
        """Test fuzzy search capabilities"""
        request = SearchRequest(
            query="programing",  # Misspelled
            search_type=SearchType.FUZZY,
            fuzzy_threshold=0.5
        )

        response = await self.client.post("/search", json=request.dict())

        assert response.status_code == 200
        data = response.json()

        # Should still find "programming" articles
        assert len(data['results']) > 0

    async def test_duplicate_detection(self):
        """Test duplicate detection endpoint"""
        # Add original article
        await self.mock_db.add_test_articles([{
            'id': '1',
            'title': 'Test Article',
            'content': 'This is a test article content for duplicate detection testing purposes',
            'url': 'https://example.com/test1',
            'content_hash': 'hash1'
        }])

        # Test with exact duplicate
        duplicate_request = {
            'title': 'Test Article',
            'content': 'This is a test article content for duplicate detection testing purposes',
            'threshold': 0.9
        }

        response = await self.client.post("/detect-duplicate", json=duplicate_request)

        assert response.status_code == 200
        data = response.json()

        assert data['is_duplicate'] == True
        assert data['confidence'] > 0.9

    async def test_search_facets(self):
        """Test search facets generation"""
        request = SearchRequest(query="python")

        response = await self.client.post("/search", json=request.dict())

        assert response.status_code == 200
        data = response.json()

        # Should have facets
        assert 'facets' in data
        assert 'languages' in data['facets']
        assert 'domains' in data['facets']

    async def test_pagination(self):
        """Test search pagination"""
        request = SearchRequest(query="python", limit=1, offset=0)

        response = await self.client.post("/search", json=request.dict())

        assert response.status_code == 200
        data = response.json()

        assert len(data['results']) <= 1
        assert 'pagination' in data
        assert data['pagination']['limit'] == 1
        assert data['pagination']['offset'] == 0

    async def test_health_check(self):
        """Test health check endpoint"""
        response = await self.client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'healthy'
        assert 'timestamp' in data


class MockDatabase:
    """Mock database for testing"""

    def __init__(self):
        self.articles = []

    async def add_test_articles(self, articles):
        """Add test articles"""
        self.articles.extend(articles)

    async def fetch(self, query, *params):
        """Mock fetch method"""
        # Simple mock - return articles that match basic criteria
        if "python" in query.lower():
            return [
                {
                    'id': '1',
                    'url': 'https://example.com/python-guide',
                    'title': 'Python Programming Guide',
                    'content': 'Learn Python programming from basics to advanced',
                    'author': 'John Doe',
                    'published_at': None,
                    'crawled_at': None,
                    'source_domain': 'example.com',
                    'language': 'en',
                    'word_count': 150,
                    'tags': ['python', 'programming'],
                    'metadata': {},
                    'score': 1.0
                }
            ]
        return []

    async def fetchrow(self, query, *params):
        """Mock fetchrow method"""
        results = await self.fetch(query, *params)
        return results[0] if results else None

    async def fetchval(self, query, *params):
        """Mock fetchval method"""
        if "SELECT 1" in query:
            return 1
        return None

    async def execute(self, query, *params):
        """Mock execute method"""
        return "MOCK EXECUTED"


class TestDuplicateDetector:
    """Test duplicate detection functionality"""

    def setup_method(self):
        self.mock_db = MockDatabase()
        self.detector = DuplicateDetector(self.mock_db)

    async def test_exact_duplicate_detection(self):
        """Test exact duplicate detection"""
        # Add test article
        await self.mock_db.add_test_articles([{
            'id': '1',
            'title': 'Test Article',
            'content': 'This is test content',
            'url': 'https://example.com/test',
            'content_hash': 'testhash'
        }])

        result = await self.detector.detect_duplicate(
            title='Test Article',
            content='This is test content'
        )

        assert result['is_duplicate'] == True
        assert result['confidence'] == 1.0

    async def test_similar_content_detection(self):
        """Test similar content detection"""
        # Test with slightly different content
        result = await self.detector.detect_duplicate(
            title='Test Article',
            content='This is test content with some additional words'
        )

        # Should not be detected as duplicate due to content differences
        assert result['is_duplicate'] == False

    async def test_batch_duplicate_detection(self):
        """Test batch duplicate detection"""
        articles = [
            {'title': 'Article 1', 'content': 'Content 1'},
            {'title': 'Article 2', 'content': 'Content 2'},
            {'title': 'Article 1', 'content': 'Content 1'}  # Duplicate
        ]

        results = await self.detector.batch_detect_duplicates(articles)

        assert len(results) == 3
        # The third article should be detected as duplicate
        assert results[2]['is_duplicate'] == True
```

## Exercises

### Exercise 1: Search API Development
1. Implement RESTful search endpoints with proper request/response models
2. Add support for multiple search types (exact, fuzzy, semantic, hybrid)
3. Implement advanced filtering and sorting capabilities
4. Add search result highlighting and pagination

### Exercise 2: Duplicate Detection Implementation
1. Create content normalization and similarity calculation algorithms
2. Implement multi-strategy duplicate detection (hash, fuzzy, semantic)
3. Build real-time duplicate prevention for crawling pipelines
4. Add batch duplicate detection for existing data

### Exercise 3: Search Optimization
1. Implement search result caching and performance monitoring
2. Add search analytics and query performance tracking
3. Optimize database queries and indexes for search operations
4. Implement search result ranking and relevance scoring

### Exercise 4: API Testing and Validation
1. Create comprehensive test suites for search and duplicate detection
2. Implement load testing for concurrent search operations
3. Add API monitoring and error handling
4. Build search result validation and quality assurance

## Next Steps
- Complete [Workshop 06: Pipeline Monitoring](../workshops/workshop-06-pipeline-monitoring.md)
- Learn about production deployment and scaling
- Explore advanced search and analytics features

## Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Text Similarity Algorithms](https://towardsdatascience.com/text-similarity-algorithms-6d3bb7753bd2)
- [API Design Best Practices](https://docs.microsoft.com/en-us/azure/architecture/best-practices/api-design)