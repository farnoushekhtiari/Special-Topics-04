# Workshop 04: Database Indexes and Performance Optimization

## Overview
This workshop focuses on implementing database indexes and optimizing query performance for crawled data storage. You'll learn how to analyze query patterns, create appropriate indexes, and monitor performance improvements in PostgreSQL.

## Prerequisites
- Completed [Indexes and Performance Tutorial](../tutorials/03-indexes-performance.md)
- Basic understanding of PostgreSQL
- Knowledge of SQL queries and joins

## Learning Objectives
By the end of this workshop, you will be able to:
- Analyze query execution plans and identify performance bottlenecks
- Create appropriate indexes for different query patterns
- Implement composite and partial indexes
- Monitor index usage and maintenance
- Optimize database performance for crawling applications

## Workshop Structure

### Part 1: Query Analysis and Index Planning

#### Step 1: Create Performance Analysis Tools

```python
# utils/query_analyzer.py
import asyncio
import time
from typing import Dict, List, Any, Optional
import asyncpg
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueryMetrics:
    """Metrics for query performance analysis"""
    query: str
    execution_time: float
    rows_affected: int
    plan: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'execution_time': self.execution_time,
            'rows_affected': self.rows_affected,
            'plan': self.plan,
            'timestamp': self.timestamp.isoformat()
        }


class QueryAnalyzer:
    """Analyzes PostgreSQL query performance"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection_pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        self.connection_pool = await asyncpg.create_pool(self.connection_string)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection_pool:
            await self.connection_pool.close()

    async def analyze_query(self, query: str, params: tuple = None) -> QueryMetrics:
        """Analyze query execution with EXPLAIN ANALYZE"""
        if not self.connection_pool:
            raise RuntimeError("Analyzer not initialized")

        async with self.connection_pool.acquire() as conn:
            # Enable timing
            await conn.execute("SET track_timing = ON")

            # Get execution plan
            explain_query = f"EXPLAIN (ANALYZE, VERBOSE, COSTS, BUFFERS, TIMING) {query}"

            start_time = time.time()
            try:
                # Execute query with analysis
                if params:
                    rows = await conn.fetch(explain_query, *params)
                else:
                    rows = await conn.fetch(explain_query)

                execution_time = time.time() - start_time

                # Parse EXPLAIN output
                plan_text = '\n'.join(row[0] for row in rows)
                plan = self._parse_explain_output(plan_text)

                # Get actual row count by re-executing without EXPLAIN
                actual_start = time.time()
                if params:
                    actual_rows = await conn.fetch(query, *params)
                else:
                    actual_rows = await conn.fetch(query)
                actual_time = time.time() - actual_start

                return QueryMetrics(
                    query=query,
                    execution_time=actual_time,
                    rows_affected=len(actual_rows),
                    plan=plan,
                    timestamp=datetime.now()
                )

            except Exception as e:
                execution_time = time.time() - start_time
                return QueryMetrics(
                    query=query,
                    execution_time=execution_time,
                    rows_affected=0,
                    plan={'error': str(e)},
                    timestamp=datetime.now()
                )

    def _parse_explain_output(self, plan_text: str) -> Dict[str, Any]:
        """Parse EXPLAIN ANALYZE output into structured format"""
        lines = plan_text.split('\n')
        parsed_plan = {
            'raw_plan': plan_text,
            'operations': [],
            'total_cost': 0,
            'actual_time': 0,
            'buffers': {}
        }

        current_operation = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse operation lines
            if '->' in line or line.startswith('Seq Scan') or line.startswith('Index Scan'):
                operation = self._parse_operation_line(line)
                if operation:
                    parsed_plan['operations'].append(operation)
                    current_operation = operation

            # Parse cost information
            cost_match = re.search(r'cost=(\d+\.\d+)\.\.(\d+\.\d+)', line)
            if cost_match:
                parsed_plan['total_cost'] = float(cost_match.group(2))

            # Parse actual time
            time_match = re.search(r'actual time=(\d+\.\d+)\.\.(\d+\.\d+)', line)
            if time_match:
                parsed_plan['actual_time'] = float(time_match.group(2))

            # Parse buffer information
            if 'Buffers:' in line:
                buffer_info = self._parse_buffer_info(line)
                parsed_plan['buffers'].update(buffer_info)

        return parsed_plan

    def _parse_operation_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single operation line from EXPLAIN"""
        # Remove arrow
        line = line.replace('->', '').strip()

        operation = {
            'type': '',
            'table': '',
            'conditions': [],
            'cost': {},
            'actual': {}
        }

        # Extract operation type
        if 'Seq Scan' in line:
            operation['type'] = 'Seq Scan'
        elif 'Index Scan' in line:
            operation['type'] = 'Index Scan'
        elif 'Index Only Scan' in line:
            operation['type'] = 'Index Only Scan'
        elif 'Bitmap Heap Scan' in line:
            operation['type'] = 'Bitmap Heap Scan'
        elif 'Hash Join' in line:
            operation['type'] = 'Hash Join'
        elif 'Nested Loop' in line:
            operation['type'] = 'Nested Loop'
        else:
            operation['type'] = line.split()[0] if line.split() else 'Unknown'

        # Extract table name
        on_match = re.search(r'on (\w+)', line)
        if on_match:
            operation['table'] = on_match.group(1)

        return operation

    def _parse_buffer_info(self, line: str) -> Dict[str, int]:
        """Parse buffer usage information"""
        buffers = {}

        # Extract buffer counts
        shared_hit_match = re.search(r'shared hit=(\d+)', line)
        if shared_hit_match:
            buffers['shared_hit'] = int(shared_hit_match.group(1))

        shared_read_match = re.search(r'shared read=(\d+)', line)
        if shared_read_match:
            buffers['shared_read'] = int(shared_read_match.group(1))

        return buffers

    async def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics for index planning"""
        if not self.connection_pool:
            raise RuntimeError("Analyzer not initialized")

        async with self.connection_pool.acquire() as conn:
            # Get table size and row count
            size_query = """
            SELECT
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_rows,
                n_dead_tup as dead_rows,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_stat_user_tables
            WHERE tablename = $1
            """

            stats = await conn.fetchrow(size_query, table_name)

            if not stats:
                return {}

            # Get column statistics
            column_query = """
            SELECT
                attname as column_name,
                n_distinct,
                correlation,
                most_common_vals,
                most_common_freqs
            FROM pg_stats
            WHERE tablename = $1
            ORDER BY attname
            """

            columns = await conn.fetch(column_query, table_name)

            return {
                'table_name': table_name,
                'size': stats['size'],
                'live_rows': stats['live_rows'],
                'dead_rows': stats['dead_rows'],
                'inserts': stats['inserts'],
                'updates': stats['updates'],
                'deletes': stats['deletes'],
                'columns': [dict(col) for col in columns]
            }

    async def suggest_indexes(self, table_name: str, query_patterns: List[str]) -> List[Dict[str, Any]]:
        """Suggest indexes based on query patterns and table statistics"""
        suggestions = []

        table_stats = await self.get_table_statistics(table_name)
        if not table_stats:
            return suggestions

        # Analyze query patterns
        for query in query_patterns:
            query_lower = query.lower()

            # Look for WHERE clauses
            where_match = re.search(r'where\s+(.+?)(?:\s+(?:group by|order by|limit|$))', query_lower, re.IGNORECASE | re.DOTALL)
            if where_match:
                where_clause = where_match.group(1)

                # Extract column names from WHERE clause
                columns = re.findall(r'(\w+)\s*[=<>!]+\s*[^\'"()\s]+', where_clause)
                columns.extend(re.findall(r'(\w+)\s+like\s+', where_clause, re.IGNORECASE))

                if columns:
                    # Suggest single column indexes
                    for col in set(columns):
                        if col in [c['column_name'] for c in table_stats['columns']]:
                            suggestions.append({
                                'type': 'single_column',
                                'table': table_name,
                                'column': col,
                                'ddl': f'CREATE INDEX idx_{table_name}_{col} ON {table_name} ({col});',
                                'reason': f'Column {col} used in WHERE clause'
                            })

                    # Suggest composite indexes for multiple columns
                    if len(set(columns)) > 1:
                        col_list = ', '.join(set(columns))
                        suggestions.append({
                            'type': 'composite',
                            'table': table_name,
                            'columns': list(set(columns)),
                            'ddl': f'CREATE INDEX idx_{table_name}_composite ON {table_name} ({col_list});',
                            'reason': f'Multiple columns used in WHERE clause: {col_list}'
                        })

            # Look for ORDER BY clauses
            order_match = re.search(r'order by\s+(.+?)(?:\s+(?:limit|$))', query_lower, re.IGNORECASE)
            if order_match:
                order_columns = [col.strip() for col in order_match.group(1).split(',')]
                for col in order_columns:
                    col_name = col.split()[0]  # Remove ASC/DESC
                    if col_name in [c['column_name'] for c in table_stats['columns']]:
                        suggestions.append({
                            'type': 'order_by',
                            'table': table_name,
                            'column': col_name,
                            'ddl': f'CREATE INDEX idx_{table_name}_{col_name}_order ON {table_name} ({col_name});',
                            'reason': f'Column {col_name} used in ORDER BY clause'
                        })

        # Remove duplicates
        seen_ddls = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion['ddl'] not in seen_ddls:
                unique_suggestions.append(suggestion)
                seen_ddls.add(suggestion['ddl'])

        return unique_suggestions
```

#### Step 2: Create Index Management System

```python
# db/index_manager.py
import asyncio
from typing import Dict, List, Any, Optional
import asyncpg
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IndexInfo:
    """Information about a database index"""
    name: str
    table: str
    columns: List[str]
    definition: str
    size: str
    is_unique: bool
    is_primary: bool
    created_at: datetime

    @classmethod
    def from_pg_row(cls, row) -> 'IndexInfo':
        return cls(
            name=row['indexname'],
            table=row['tablename'],
            columns=row['columns'],
            definition=row['indexdef'],
            size=row['size'],
            is_unique=row['is_unique'],
            is_primary=row['is_primary'],
            created_at=row['created_at']
        )


class IndexManager:
    """Manages database indexes for performance optimization"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection_pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        self.connection_pool = await asyncpg.create_pool(self.connection_string)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection_pool:
            await self.connection_pool.close()

    async def list_indexes(self, table_name: Optional[str] = None) -> List[IndexInfo]:
        """List all indexes in the database"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        async with self.connection_pool.acquire() as conn:
            query = """
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef,
                pg_size_pretty(pg_relation_size(indexrelid)) as size,
                indisunique as is_unique,
                indisprimary as is_primary,
                obj_description(indexrelid, 'pg_class') as comment,
                idx.indkey,
                array_agg(att.attname ORDER BY array_position(idx.indkey, att.attnum)) as columns
            FROM pg_index idx
            JOIN pg_class tbl ON tbl.oid = idx.indrelid
            JOIN pg_class idx_cls ON idx_cls.oid = idx.indexrelid
            LEFT JOIN pg_attribute att ON att.attrelid = tbl.oid AND att.attnum = ANY(idx.indkey)
            WHERE schemaname = 'public'
                AND idx_cls.relkind = 'i'
                {table_filter}
            GROUP BY schemaname, tablename, indexname, indexdef, idx.indrelid, idx.indexrelid,
                     is_unique, is_primary, comment, idx.indkey
            ORDER BY tablename, indexname
            """

            table_filter = f"AND tablename = '{table_name}'" if table_name else ""

            rows = await conn.fetch(query.format(table_filter=table_filter))

            indexes = []
            for row in rows:
                # Get creation time (approximate)
                created_at = datetime.now()  # This is approximate

                index_info = IndexInfo.from_pg_row({
                    'indexname': row['indexname'],
                    'tablename': row['tablename'],
                    'columns': row['columns'],
                    'indexdef': row['indexdef'],
                    'size': row['size'],
                    'is_unique': row['is_unique'],
                    'is_primary': row['is_primary'],
                    'created_at': created_at
                })
                indexes.append(index_info)

            return indexes

    async def create_index(self, table: str, columns: List[str], index_name: Optional[str] = None,
                          unique: bool = False, concurrently: bool = True) -> str:
        """Create a new index"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        if not index_name:
            col_str = '_'.join(columns)
            index_name = f"idx_{table}_{col_str}"

        unique_str = "UNIQUE" if unique else ""
        concurrently_str = "CONCURRENTLY" if concurrently else ""
        col_str = ', '.join(columns)

        ddl = f"CREATE {unique_str} INDEX {concurrently_str} {index_name} ON {table} ({col_str})"

        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(ddl)
                return index_name
            except Exception as e:
                raise Exception(f"Failed to create index: {e}")

    async def drop_index(self, index_name: str, concurrently: bool = True) -> bool:
        """Drop an index"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        concurrently_str = "CONCURRENTLY" if concurrently else ""
        ddl = f"DROP INDEX {concurrently_str} {index_name}"

        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(ddl)
                return True
            except Exception as e:
                print(f"Failed to drop index {index_name}: {e}")
                return False

    async def get_unused_indexes(self) -> List[IndexInfo]:
        """Find indexes that haven't been used recently"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        async with self.connection_pool.acquire() as conn:
            # Get index usage statistics
            query = """
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan as scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as size
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
                AND idx_scan = 0
            ORDER BY pg_relation_size(indexrelid) DESC
            """

            rows = await conn.fetch(query)

            unused_indexes = []
            for row in rows:
                # Get index details
                index_info = await self._get_index_details(row['indexname'])
                if index_info:
                    unused_indexes.append(index_info)

            return unused_indexes

    async def _get_index_details(self, index_name: str) -> Optional[IndexInfo]:
        """Get detailed information about a specific index"""
        indexes = await self.list_indexes()
        for idx in indexes:
            if idx.name == index_name:
                return idx
        return None

    async def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Run ANALYZE on a table to update statistics"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        async with self.connection_pool.acquire() as conn:
            await conn.execute(f"ANALYZE {table_name}")

            # Get updated statistics
            stats_query = """
            SELECT
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_rows,
                n_dead_tup as dead_rows,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            WHERE tablename = $1
            """

            stats = await conn.fetchrow(stats_query, table_name)
            return dict(stats) if stats else {}

    async def reindex_table(self, table_name: str, concurrently: bool = True) -> bool:
        """Rebuild indexes for a table"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        concurrently_str = "CONCURRENTLY" if concurrently else ""
        ddl = f"REINDEX {concurrently_str} TABLE {table_name}"

        async with self.connection_pool.acquire() as conn:
            try:
                await conn.execute(ddl)
                return True
            except Exception as e:
                print(f"Failed to reindex table {table_name}: {e}")
                return False

    async def get_index_usage_report(self) -> List[Dict[str, Any]]:
        """Generate index usage report"""
        if not self.connection_pool:
            raise RuntimeError("Index manager not initialized")

        async with self.connection_pool.acquire() as conn:
            query = """
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan as scans,
                idx_tup_read as tuples_read,
                idx_tup_fetch as tuples_fetched,
                pg_size_pretty(pg_relation_size(indexrelid)) as size,
                CASE
                    WHEN idx_scan > 0 THEN pg_size_pretty(pg_relation_size(indexrelid) / idx_scan)
                    ELSE 'Unused'
                END as cost_per_scan
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC, pg_relation_size(indexrelid) DESC
            """

            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def optimize_indexes(self, table_name: str) -> List[str]:
        """Analyze and suggest index optimizations"""
        recommendations = []

        # Get current indexes
        indexes = await self.list_indexes(table_name)

        # Get table statistics
        table_stats = await self.analyze_table(table_name)

        # Check for unused indexes
        unused = await self.get_unused_indexes()
        unused_names = [idx.name for idx in unused if idx.table == table_name]

        for unused_name in unused_names:
            recommendations.append(f"Consider dropping unused index: {unused_name}")

        # Check index bloat (simplified)
        if table_stats.get('dead_rows', 0) > table_stats.get('live_rows', 1) * 0.2:
            recommendations.append(f"Table {table_name} has high dead tuple ratio, consider VACUUM")

        # Suggest REINDEX for large tables
        if table_stats.get('live_rows', 0) > 100000:
            recommendations.append(f"Consider REINDEX for large table: {table_name}")

        return recommendations
```

### Part 2: Performance Optimization Implementation

#### Step 1: Create Crawled Data Schema with Indexes

```sql
-- Create optimized schema for crawled data with indexes
CREATE TABLE IF NOT EXISTS crawled_articles (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    content TEXT,
    summary TEXT,
    author TEXT,
    published_at TIMESTAMP,
    crawled_at TIMESTAMP DEFAULT NOW(),
    content_hash CHAR(64) UNIQUE,  -- For deduplication
    word_count INTEGER,
    language VARCHAR(10),
    source_domain TEXT,
    tags TEXT[],  -- Array of tags
    metadata JSONB,  -- Flexible metadata storage
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Essential indexes for crawling operations
CREATE INDEX IF NOT EXISTS idx_crawled_articles_url ON crawled_articles (url);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_content_hash ON crawled_articles (content_hash);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_crawled_at ON crawled_articles (crawled_at);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_published_at ON crawled_articles (published_at);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_source_domain ON crawled_articles (source_domain);

-- Performance indexes for common queries
CREATE INDEX IF NOT EXISTS idx_crawled_articles_language ON crawled_articles (language);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_author ON crawled_articles (author);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_word_count ON crawled_articles (word_count);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_crawled_articles_content_fts ON crawled_articles USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_crawled_articles_title_fts ON crawled_articles USING GIN (to_tsvector('english', title));

-- JSONB indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_crawled_articles_metadata ON crawled_articles USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_tags ON crawled_articles USING GIN (tags);

-- Composite indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_crawled_articles_domain_date ON crawled_articles (source_domain, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_author_date ON crawled_articles (author, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_crawled_articles_lang_date ON crawled_articles (language, crawled_at DESC);

-- Partial indexes for specific use cases
CREATE INDEX IF NOT EXISTS idx_recent_articles ON crawled_articles (crawled_at DESC) WHERE crawled_at > NOW() - INTERVAL '30 days';
CREATE INDEX IF NOT EXISTS idx_popular_authors ON crawled_articles (author) WHERE author IS NOT NULL AND word_count > 100;
```

#### Step 2: Create Performance Testing Suite

```python
# tests/performance_tests.py
import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
import asyncpg
from utils.query_analyzer import QueryAnalyzer
from db.index_manager import IndexManager


class PerformanceTestSuite:
    """Comprehensive performance testing for database operations"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        self.pool = await asyncpg.create_pool(self.connection_string)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()

    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        results = {
            'timestamp': time.time(),
            'tests': {}
        }

        print("Running performance test suite...")

        # Test data insertion performance
        results['tests']['insert_performance'] = await self.test_insert_performance()

        # Test query performance
        results['tests']['query_performance'] = await self.test_query_performance()

        # Test index effectiveness
        results['tests']['index_effectiveness'] = await self.test_index_effectiveness()

        # Test concurrent operations
        results['tests']['concurrency_performance'] = await self.test_concurrency_performance()

        # Test memory usage
        results['tests']['memory_usage'] = await self.test_memory_usage()

        return results

    async def test_insert_performance(self) -> Dict[str, Any]:
        """Test bulk insert performance"""
        print("Testing insert performance...")

        # Create test data
        test_articles = self._generate_test_articles(1000)

        # Test without indexes (temporarily drop some)
        await self._drop_performance_indexes()
        start_time = time.time()
        await self._bulk_insert_articles(test_articles)
        no_index_time = time.time() - start_time

        # Clear test data
        await self._clear_test_data()

        # Recreate indexes
        await self._create_performance_indexes()

        # Test with indexes
        start_time = time.time()
        await self._bulk_insert_articles(test_articles)
        with_index_time = time.time() - start_time

        return {
            'no_index_time': no_index_time,
            'with_index_time': with_index_time,
            'slowdown_ratio': with_index_time / no_index_time if no_index_time > 0 else 0,
            'articles_per_second_no_index': len(test_articles) / no_index_time if no_index_time > 0 else 0,
            'articles_per_second_with_index': len(test_articles) / with_index_time if with_index_time > 0 else 0
        }

    async def test_query_performance(self) -> Dict[str, Any]:
        """Test various query performance scenarios"""
        print("Testing query performance...")

        async with QueryAnalyzer(self.connection_string) as analyzer:
            queries = [
                "SELECT COUNT(*) FROM crawled_articles",
                "SELECT * FROM crawled_articles WHERE source_domain = 'example.com' LIMIT 100",
                "SELECT * FROM crawled_articles WHERE published_at > NOW() - INTERVAL '7 days'",
                "SELECT * FROM crawled_articles WHERE author IS NOT NULL ORDER BY published_at DESC LIMIT 50",
                "SELECT * FROM crawled_articles WHERE content @@ to_tsquery('english', 'python & programming')",
                "SELECT source_domain, COUNT(*) FROM crawled_articles GROUP BY source_domain ORDER BY COUNT(*) DESC LIMIT 10"
            ]

            results = {}
            for query in queries:
                metrics = await analyzer.analyze_query(query)
                query_key = query.split()[1] if len(query.split()) > 1 else 'complex'
                results[query_key] = {
                    'execution_time': metrics.execution_time,
                    'rows_affected': metrics.rows_affected,
                    'plan_summary': metrics.plan
                }

            return results

    async def test_index_effectiveness(self) -> Dict[str, Any]:
        """Test how well indexes improve query performance"""
        print("Testing index effectiveness...")

        async with QueryAnalyzer(self.connection_string) as analyzer:
            # Test queries that should benefit from indexes
            test_queries = [
                ("domain_query", "SELECT * FROM crawled_articles WHERE source_domain = 'test.com'"),
                ("date_query", "SELECT * FROM crawled_articles WHERE published_at > NOW() - INTERVAL '1 day'"),
                ("author_query", "SELECT * FROM crawled_articles WHERE author = 'Test Author'"),
                ("fts_query", "SELECT * FROM crawled_articles WHERE content @@ to_tsquery('english', 'test')"),
                ("composite_query", "SELECT * FROM crawled_articles WHERE source_domain = 'test.com' AND published_at > NOW() - INTERVAL '1 day'")
            ]

            results = {}
            for query_name, query in test_queries:
                metrics = await analyzer.analyze_query(query)
                results[query_name] = {
                    'execution_time': metrics.execution_time,
                    'uses_index': self._check_if_uses_index(metrics.plan),
                    'buffer_usage': metrics.plan.get('buffers', {})
                }

            return results

    async def test_concurrency_performance(self) -> Dict[str, Any]:
        """Test performance under concurrent load"""
        print("Testing concurrency performance...")

        concurrency_levels = [1, 5, 10, 20]
        results = {}

        for concurrency in concurrency_levels:
            print(f"Testing with {concurrency} concurrent connections...")

            start_time = time.time()
            tasks = []

            for i in range(concurrency):
                task = asyncio.create_task(self._run_concurrent_queries())
                tasks.append(task)

            await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            results[f'concurrency_{concurrency}'] = {
                'total_time': total_time,
                'avg_time_per_query': total_time / (concurrency * 10),  # 10 queries per connection
                'queries_per_second': (concurrency * 10) / total_time if total_time > 0 else 0
            }

        return results

    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        print("Testing memory usage...")

        if not self.pool:
            return {}

        async with self.pool.acquire() as conn:
            # Get database size
            size_result = await conn.fetchrow("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as db_size,
                       pg_size_pretty(sum(pg_relation_size(oid))) as table_size
                FROM pg_class
                WHERE relkind IN ('r', 'i')
            """)

            # Get cache hit ratio
            cache_result = await conn.fetchrow("""
                SELECT
                    sum(blks_hit) * 100.0 / (sum(blks_hit) + sum(blks_read)) as cache_hit_ratio
                FROM pg_stat_database
                WHERE datname = current_database()
            """)

            return {
                'database_size': size_result['db_size'] if size_result else 'unknown',
                'table_size': size_result['table_size'] if size_result else 'unknown',
                'cache_hit_ratio': cache_result['cache_hit_ratio'] if cache_result and cache_result['cache_hit_ratio'] else 0
            }

    async def _run_concurrent_queries(self) -> None:
        """Run a set of queries for concurrency testing"""
        if not self.pool:
            return

        queries = [
            "SELECT COUNT(*) FROM crawled_articles",
            "SELECT * FROM crawled_articles WHERE source_domain = 'example.com' LIMIT 10",
            "SELECT source_domain, COUNT(*) FROM crawled_articles GROUP BY source_domain LIMIT 5",
            "SELECT * FROM crawled_articles ORDER BY crawled_at DESC LIMIT 20",
            "SELECT * FROM crawled_articles WHERE published_at > NOW() - INTERVAL '1 day' LIMIT 15",
            "SELECT DISTINCT author FROM crawled_articles WHERE author IS NOT NULL LIMIT 10",
            "SELECT * FROM crawled_articles WHERE word_count > 500 LIMIT 10",
            "SELECT source_domain, AVG(word_count) FROM crawled_articles GROUP BY source_domain LIMIT 5",
            "SELECT * FROM crawled_articles WHERE tags @> ARRAY['technology'] LIMIT 10",
            "SELECT * FROM crawled_articles WHERE metadata->>'language' = 'en' LIMIT 10"
        ]

        async with self.pool.acquire() as conn:
            for query in queries:
                try:
                    await conn.fetch(query)
                except Exception:
                    pass  # Ignore errors in performance test

    def _generate_test_articles(self, count: int) -> List[Dict[str, Any]]:
        """Generate test article data"""
        import random
        from datetime import datetime, timedelta

        domains = ['example.com', 'news.com', 'tech.com', 'blog.com']
        authors = ['Alice', 'Bob', 'Charlie', 'Diana', None]
        languages = ['en', 'fa', 'ar']

        articles = []
        for i in range(count):
            published_at = datetime.now() - timedelta(days=random.randint(0, 365))
            articles.append({
                'url': f"https://{random.choice(domains)}/article/{i}",
                'title': f"Test Article {i}",
                'content': f"This is test content for article {i}. " * random.randint(10, 50),
                'author': random.choice(authors),
                'published_at': published_at,
                'crawled_at': datetime.now(),
                'content_hash': f"hash_{i}",
                'word_count': random.randint(100, 2000),
                'language': random.choice(languages),
                'source_domain': random.choice(domains),
                'tags': random.sample(['tech', 'news', 'blog', 'tutorial', 'review'], random.randint(0, 3)),
                'metadata': {'test': True, 'batch': i // 100}
            })

        return articles

    async def _bulk_insert_articles(self, articles: List[Dict[str, Any]]) -> None:
        """Bulk insert articles"""
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO crawled_articles
                (url, title, content, author, published_at, crawled_at,
                 content_hash, word_count, language, source_domain, tags, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (url) DO NOTHING
            """, [(
                art['url'], art['title'], art['content'], art['author'],
                art['published_at'], art['crawled_at'], art['content_hash'],
                art['word_count'], art['language'], art['source_domain'],
                art['tags'], json.dumps(art['metadata'])
            ) for art in articles])

    async def _clear_test_data(self) -> None:
        """Clear test data"""
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM crawled_articles WHERE metadata->>'test' = 'true'")

    async def _drop_performance_indexes(self) -> None:
        """Temporarily drop performance indexes for testing"""
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            indexes_to_drop = [
                'idx_crawled_articles_published_at',
                'idx_crawled_articles_author',
                'idx_crawled_articles_word_count',
                'idx_crawled_articles_tags',
                'idx_crawled_articles_domain_date'
            ]

            for index in indexes_to_drop:
                try:
                    await conn.execute(f"DROP INDEX IF EXISTS {index}")
                except Exception:
                    pass  # Ignore errors

    async def _create_performance_indexes(self) -> None:
        """Recreate performance indexes"""
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_articles_published_at ON crawled_articles (published_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_articles_author ON crawled_articles (author)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_articles_word_count ON crawled_articles (word_count)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_articles_tags ON crawled_articles USING GIN (tags)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_articles_domain_date ON crawled_articles (source_domain, published_at DESC)")

    def _check_if_uses_index(self, plan: Dict[str, Any]) -> bool:
        """Check if query plan uses indexes"""
        operations = plan.get('operations', [])
        for op in operations:
            if 'Index' in op.get('type', ''):
                return True
        return False
```

### Part 3: Index Maintenance and Monitoring

#### Step 1: Create Index Maintenance Tools

```python
# maintenance/index_maintenance.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from db.index_manager import IndexManager
from utils.query_analyzer import QueryAnalyzer


class IndexMaintenanceScheduler:
    """Automated index maintenance and optimization"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger('IndexMaintenance')
        self.maintenance_history: List[Dict[str, Any]] = []

    async def run_full_maintenance(self) -> Dict[str, Any]:
        """Run complete index maintenance routine"""
        start_time = datetime.now()

        results = {
            'start_time': start_time,
            'tasks_completed': [],
            'errors': [],
            'recommendations': []
        }

        try:
            # Analyze unused indexes
            unused_indexes = await self._analyze_unused_indexes()
            if unused_indexes:
                results['recommendations'].extend([
                    f"Consider dropping unused index: {idx.name} on {idx.table}"
                    for idx in unused_indexes
                ])

            # Check index bloat
            bloated_indexes = await self._check_index_bloat()
            if bloated_indexes:
                results['tasks_completed'].append('index_bloat_check')
                results['recommendations'].extend([
                    f"Rebuild bloated index: {idx['name']} (bloat: {idx['bloat_ratio']:.1%})"
                    for idx in bloated_indexes
                ])

            # Update table statistics
            await self._update_table_statistics()
            results['tasks_completed'].append('statistics_update')

            # Reindex large tables if needed
            reindexed_tables = await self._reindex_large_tables()
            if reindexed_tables:
                results['tasks_completed'].append('table_reindex')
                results['tasks_completed'].extend([f"reindexed_{table}" for table in reindexed_tables])

            # Optimize query performance
            optimizations = await self._optimize_query_performance()
            results['tasks_completed'].extend(optimizations)

        except Exception as e:
            results['errors'].append(str(e))
            self.logger.error(f"Maintenance failed: {e}")

        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - start_time).total_seconds()

        self.maintenance_history.append(results)
        return results

    async def _analyze_unused_indexes(self) -> List[Any]:
        """Find indexes that haven't been used recently"""
        async with IndexManager(self.connection_string) as manager:
            return await manager.get_unused_indexes()

    async def _check_index_bloat(self) -> List[Dict[str, Any]]:
        """Check for bloated indexes that need rebuilding"""
        bloated_indexes = []

        # This is a simplified bloat check
        # In production, you'd use more sophisticated bloat detection
        async with IndexManager(self.connection_string) as manager:
            indexes = await manager.list_indexes()

            for index in indexes:
                # Simple heuristic: if index is large and table has been heavily modified
                # In practice, you'd use pgstattuple or similar extensions
                if 'MB' in index.size and float(index.size.replace(' MB', '')) > 100:
                    bloated_indexes.append({
                        'name': index.name,
                        'table': index.table,
                        'size': index.size,
                        'bloat_ratio': 0.3  # Estimated bloat ratio
                    })

        return bloated_indexes

    async def _update_table_statistics(self) -> None:
        """Update table statistics for query planner"""
        async with IndexManager(self.connection_string) as manager:
            # Analyze all tables
            async with manager.connection_pool.acquire() as conn:
                tables = await conn.fetch("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                """)

                for table in tables:
                    await conn.execute(f"ANALYZE {table['tablename']}")

    async def _reindex_large_tables(self) -> List[str]:
        """Reindex tables that are large or heavily modified"""
        reindexed_tables = []

        async with IndexManager(self.connection_string) as manager:
            async with manager.connection_pool.acquire() as conn:
                # Find large tables
                large_tables = await conn.fetch("""
                    SELECT tablename,
                           pg_size_pretty(pg_total_relation_size(tablename)) as size
                    FROM pg_tables
                    WHERE schemaname = 'public'
                        AND pg_total_relation_size(tablename) > 100 * 1024 * 1024  -- 100MB
                """)

                for table in large_tables:
                    table_name = table['tablename']
                    try:
                        success = await manager.reindex_table(table_name)
                        if success:
                            reindexed_tables.append(table_name)
                            self.logger.info(f"Reindexed table: {table_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to reindex {table_name}: {e}")

        return reindexed_tables

    async def _optimize_query_performance(self) -> List[str]:
        """Apply query performance optimizations"""
        optimizations = []

        async with QueryAnalyzer(self.connection_string) as analyzer:
            # Analyze slow queries and suggest optimizations
            # This is a simplified version - in practice you'd analyze query logs

            # Check if we need additional indexes based on common query patterns
            common_queries = [
                "SELECT * FROM crawled_articles WHERE source_domain = $1 ORDER BY published_at DESC LIMIT 50",
                "SELECT * FROM crawled_articles WHERE author = $1 AND published_at > $2",
                "SELECT COUNT(*) FROM crawled_articles WHERE language = $1 AND published_at > $2"
            ]

            for table in ['crawled_articles']:
                suggestions = await analyzer.suggest_indexes(table, common_queries)

                async with IndexManager(self.connection_string) as manager:
                    for suggestion in suggestions:
                        try:
                            # Check if index already exists
                            existing_indexes = await manager.list_indexes(table)
                            existing_names = [idx.name for idx in existing_indexes]

                            if suggestion['name'] not in existing_names:
                                index_name = await manager.create_index(
                                    suggestion['table'],
                                    suggestion['columns']
                                )
                                optimizations.append(f"created_index_{index_name}")
                                self.logger.info(f"Created index: {index_name}")

                        except Exception as e:
                            self.logger.error(f"Failed to create index: {e}")

        return optimizations

    async def schedule_maintenance(self, interval_hours: int = 24) -> None:
        """Schedule regular maintenance tasks"""
        while True:
            try:
                self.logger.info("Starting scheduled maintenance...")
                results = await self.run_full_maintenance()

                if results['errors']:
                    self.logger.error(f"Maintenance completed with errors: {results['errors']}")
                else:
                    self.logger.info(f"Maintenance completed successfully. Tasks: {results['tasks_completed']}")

            except Exception as e:
                self.logger.error(f"Scheduled maintenance failed: {e}")

            # Wait for next interval
            await asyncio.sleep(interval_hours * 3600)

    def get_maintenance_history(self) -> List[Dict[str, Any]]:
        """Get maintenance execution history"""
        return self.maintenance_history.copy()

    async def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive maintenance report"""
        report = {
            'generated_at': datetime.now(),
            'total_maintenance_runs': len(self.maintenance_history),
            'last_run': None,
            'average_duration': 0,
            'most_common_recommendations': {},
            'success_rate': 0
        }

        if self.maintenance_history:
            last_run = self.maintenance_history[-1]
            report['last_run'] = last_run

            durations = [run['duration'] for run in self.maintenance_history]
            report['average_duration'] = sum(durations) / len(durations)

            # Count successful runs
            successful_runs = sum(1 for run in self.maintenance_history if not run['errors'])
            report['success_rate'] = successful_runs / len(self.maintenance_history)

            # Most common recommendations
            from collections import Counter
            all_recommendations = []
            for run in self.maintenance_history:
                all_recommendations.extend(run['recommendations'])

            recommendation_counts = Counter(all_recommendations)
            report['most_common_recommendations'] = dict(recommendation_counts.most_common(5))

        return report
```

## Exercises

### Exercise 1: Index Analysis and Creation
1. Analyze query patterns in your crawling application
2. Create appropriate indexes for common query types
3. Test query performance before and after index creation
4. Monitor index usage with PostgreSQL statistics

### Exercise 2: Performance Optimization
1. Identify slow queries using EXPLAIN ANALYZE
2. Implement composite indexes for multi-column queries
3. Optimize index order for query patterns
4. Test the impact of index changes on insert performance

### Exercise 3: Index Maintenance
1. Set up automated index maintenance routines
2. Identify and remove unused indexes
3. Monitor index bloat and rebuild when necessary
4. Schedule regular REINDEX operations for large tables

### Exercise 4: Query Optimization
1. Rewrite complex queries to use indexes effectively
2. Implement partial indexes for specific use cases
3. Use appropriate index types (B-tree, GIN, GiST) for different data types
4. Optimize full-text search queries with proper indexes

## Next Steps
- Complete [Workshop 05: Storage and Persistence](../workshops/workshop-05-storage-persistence.md)
- Learn about async database connections
- Explore Alembic for database migrations

## Resources
- [PostgreSQL Index Types](https://www.postgresql.org/docs/current/indexes-types.html)
- [Query Performance Tuning](https://www.postgresql.org/docs/current/performance-tips.html)
- [Index Maintenance Best Practices](https://www.postgresql.org/docs/current/routine-vacuuming.html)
- [EXPLAIN Documentation](https://www.postgresql.org/docs/current/using-explain.html)