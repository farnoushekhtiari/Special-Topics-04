# Workshop 05: Storage Options and Data Persistence

## Overview
This workshop covers comprehensive data storage and persistence strategies for crawled data. You'll implement multiple storage backends, handle data deduplication, create indexing strategies, and build robust data pipelines that ensure data integrity and performance.

## Prerequisites
- Completed [Storage Options Tutorial](../tutorials/05-storage-options.md)
- Knowledge of database concepts and file I/O
- Understanding of data serialization formats

## Learning Objectives
By the end of this workshop, you will be able to:
- Implement multiple storage backends (JSON, databases, cloud storage)
- Handle data deduplication and integrity
- Create efficient indexing and search capabilities
- Build data backup and recovery systems
- Optimize storage performance and scalability

## Workshop Structure

### Part 1: Storage Backend Implementation

#### Step 1: Create Storage Interface and Base Classes

```python
# storage/storage_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio


@dataclass
class StorageConfig:
    """Configuration for storage backends"""
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    compression: bool = False
    encryption: bool = False
    backup_enabled: bool = True


@dataclass
class StorageItem:
    """Represents a stored item with metadata"""
    id: str
    data: Dict[str, Any]
    url: str
    crawled_at: datetime
    content_hash: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'data': self.data,
            'url': self.url,
            'crawled_at': self.crawled_at.isoformat(),
            'content_hash': self.content_hash,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageItem':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            data=data['data'],
            url=data['url'],
            crawled_at=datetime.fromisoformat(data['crawled_at']),
            content_hash=data['content_hash'],
            metadata=data.get('metadata', {})
        )


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()

    @abstractmethod
    async def store(self, item: StorageItem) -> bool:
        """Store a single item"""
        pass

    @abstractmethod
    async def store_batch(self, items: List[StorageItem]) -> int:
        """Store multiple items efficiently"""
        pass

    @abstractmethod
    async def retrieve(self, item_id: str) -> Optional[StorageItem]:
        """Retrieve a single item by ID"""
        pass

    @abstractmethod
    async def retrieve_batch(self, item_ids: List[str]) -> List[StorageItem]:
        """Retrieve multiple items by IDs"""
        pass

    @abstractmethod
    async def search(self, query: Dict[str, Any], limit: int = 100) -> List[StorageItem]:
        """Search items with query parameters"""
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID"""
        pass

    @abstractmethod
    async def exists(self, item_id: str) -> bool:
        """Check if item exists"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of stored items"""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all stored items (return count of deleted items)"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass

    @abstractmethod
    async def optimize(self) -> bool:
        """Optimize storage (rebuild indexes, vacuum, etc.)"""
        pass

    @abstractmethod
    async def backup(self, backup_path: str) -> bool:
        """Create backup of storage"""
        pass

    @abstractmethod
    async def restore(self, backup_path: str) -> bool:
        """Restore from backup"""
        pass

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    @abstractmethod
    async def initialize(self):
        """Initialize storage backend"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass


class DeduplicationMixin:
    """Mixin for handling data deduplication"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_hashes: set = set()

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if content hash has been seen before"""
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

    def generate_content_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash for content deduplication"""
        import hashlib
        import json

        # Create normalized JSON string for consistent hashing
        normalized_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(normalized_data.encode('utf-8')).hexdigest()

    async def store_with_deduplication(self, item: StorageItem) -> tuple[bool, bool]:
        """Store item with deduplication check

        Returns:
            (stored_successfully, was_duplicate)
        """
        if self.is_duplicate(item.content_hash):
            return True, True  # Successfully handled (not stored), but was duplicate

        success = await self.store(item)
        return success, False


class CompressionMixin:
    """Mixin for data compression"""

    def compress_data(self, data: str) -> bytes:
        """Compress data using gzip"""
        import gzip
        return gzip.compress(data.encode('utf-8'))

    def decompress_data(self, compressed_data: bytes) -> str:
        """Decompress gzipped data"""
        import gzip
        return gzip.decompress(compressed_data).decode('utf-8')

    def should_compress(self, data: str, threshold: int = 1024) -> bool:
        """Check if data should be compressed"""
        return len(data.encode('utf-8')) > threshold
```

#### Step 2: Implement JSON File Storage

```python
# storage/json_storage.py
import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Iterator
from pathlib import Path
import aiofiles
from storage.storage_interface import StorageBackend, StorageItem, DeduplicationMixin, CompressionMixin


class JSONFileStorage(StorageBackend, DeduplicationMixin, CompressionMixin):
    """JSON file-based storage backend"""

    def __init__(self, base_dir: str = "data/json_storage", config: StorageConfig = None):
        super().__init__(config)
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "index.json"
        self.data_dir = self.base_dir / "data"
        self.index: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def initialize(self):
        """Initialize storage directories and load index"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.index_file.exists():
            async with aiofiles.open(self.index_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                self.index = json.loads(content)

    async def cleanup(self):
        """Save index and cleanup"""
        await self._save_index()

    async def _save_index(self):
        """Save index to file"""
        async with self.lock:
            async with aiofiles.open(self.index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.index, ensure_ascii=False, indent=2))

    def _get_data_file(self, item_id: str) -> Path:
        """Get data file path for item"""
        # Simple sharding by first 2 characters of ID
        shard = item_id[:2] if len(item_id) >= 2 else "00"
        shard_dir = self.data_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{item_id}.json"

    async def store(self, item: StorageItem) -> bool:
        """Store a single item"""
        try:
            # Check for duplicates
            is_duplicate = self.is_duplicate(item.content_hash)
            if is_duplicate:
                return True  # Consider duplicate handling as successful

            data_file = self._get_data_file(item.id)

            # Prepare data for storage
            item_dict = item.to_dict()
            json_data = json.dumps(item_dict, ensure_ascii=False, indent=2)

            # Compress if enabled and beneficial
            if self.config.compression and self.should_compress(json_data):
                compressed_data = self.compress_data(json_data)
                data_file = data_file.with_suffix('.json.gz')

                async with aiofiles.open(data_file, 'wb') as f:
                    await f.write(compressed_data)
            else:
                async with aiofiles.open(data_file, 'w', encoding='utf-8') as f:
                    await f.write(json_data)

            # Update index
            async with self.lock:
                self.index[item.id] = {
                    'file_path': str(data_file.relative_to(self.base_dir)),
                    'compressed': self.config.compression and self.should_compress(json_data),
                    'size': len(json_data.encode('utf-8')),
                    'stored_at': item.crawled_at.isoformat()
                }

            return True

        except Exception as e:
            print(f"Error storing item {item.id}: {e}")
            return False

    async def store_batch(self, items: List[StorageItem]) -> int:
        """Store multiple items efficiently"""
        stored_count = 0

        # Group items by shard for batch processing
        shard_groups: Dict[str, List[StorageItem]] = {}

        for item in items:
            shard = item.id[:2] if len(item.id) >= 2 else "00"
            if shard not in shard_groups:
                shard_groups[shard] = []
            shard_groups[shard].append(item)

        # Process each shard
        for shard, shard_items in shard_groups.items():
            shard_stored = await self._store_shard_batch(shard, shard_items)
            stored_count += shard_stored

        # Save index after batch
        await self._save_index()

        return stored_count

    async def _store_shard_batch(self, shard: str, items: List[StorageItem]) -> int:
        """Store batch of items in a single shard"""
        stored_count = 0

        for item in items:
            success = await self.store(item)
            if success:
                stored_count += 1

        return stored_count

    async def retrieve(self, item_id: str) -> Optional[StorageItem]:
        """Retrieve a single item by ID"""
        if item_id not in self.index:
            return None

        try:
            index_entry = self.index[item_id]
            file_path = self.base_dir / index_entry['file_path']

            if not file_path.exists():
                return None

            # Read data
            if index_entry.get('compressed', False):
                async with aiofiles.open(file_path, 'rb') as f:
                    compressed_data = await f.read()
                json_data = self.decompress_data(compressed_data)
            else:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    json_data = await f.read()

            item_dict = json.loads(json_data)
            return StorageItem.from_dict(item_dict)

        except Exception as e:
            print(f"Error retrieving item {item_id}: {e}")
            return None

    async def retrieve_batch(self, item_ids: List[str]) -> List[StorageItem]:
        """Retrieve multiple items by IDs"""
        tasks = [self.retrieve(item_id) for item_id in item_ids]
        results = await asyncio.gather(*tasks)
        return [item for item in results if item is not None]

    async def search(self, query: Dict[str, Any], limit: int = 100) -> List[StorageItem]:
        """Search items with query parameters"""
        # Simple implementation - load all and filter
        # In production, you'd want a proper search index
        matching_items = []

        for item_id, index_entry in self.index.items():
            # Load item
            item = await self.retrieve(item_id)
            if not item:
                continue

            # Check if item matches query
            if self._matches_query(item, query):
                matching_items.append(item)

                if len(matching_items) >= limit:
                    break

        return matching_items

    def _matches_query(self, item: StorageItem, query: Dict[str, Any]) -> bool:
        """Check if item matches search query"""
        for key, value in query.items():
            if key in item.data:
                item_value = item.data[key]
                if isinstance(item_value, str) and isinstance(value, str):
                    if value.lower() not in item_value.lower():
                        return False
                elif item_value != value:
                    return False
            elif key == 'url_contains':
                if value.lower() not in item.url.lower():
                    return False
            elif key == 'date_after':
                if item.crawled_at < value:
                    return False
            elif key == 'date_before':
                if item.crawled_at > value:
                    return False

        return True

    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID"""
        if item_id not in self.index:
            return False

        try:
            index_entry = self.index[item_id]
            file_path = self.base_dir / index_entry['file_path']

            # Delete file
            if file_path.exists():
                file_path.unlink()

            # Remove from index
            async with self.lock:
                del self.index[item_id]

            return True

        except Exception as e:
            print(f"Error deleting item {item_id}: {e}")
            return False

    async def exists(self, item_id: str) -> bool:
        """Check if item exists"""
        return item_id in self.index

    async def count(self) -> int:
        """Get total number of stored items"""
        return len(self.index)

    async def clear(self) -> int:
        """Clear all stored items"""
        deleted_count = 0

        # Delete all data files
        for item_id in list(self.index.keys()):
            success = await self.delete(item_id)
            if success:
                deleted_count += 1

        return deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = 0
        compressed_count = 0

        for index_entry in self.index.values():
            total_size += index_entry.get('size', 0)
            if index_entry.get('compressed', False):
                compressed_count += 1

        return {
            'total_items': len(self.index),
            'total_size_bytes': total_size,
            'compressed_items': compressed_count,
            'compression_ratio': compressed_count / len(self.index) if self.index else 0,
            'storage_path': str(self.base_dir)
        }

    async def optimize(self) -> bool:
        """Optimize storage (rebuild index consistency)"""
        try:
            # Check for orphaned files
            actual_files = set()
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(('.json', '.json.gz')):
                        rel_path = Path(root) / file
                        actual_files.add(str(rel_path.relative_to(self.base_dir)))

            indexed_files = set()
            for index_entry in self.index.values():
                indexed_files.add(index_entry['file_path'])

            # Remove orphaned index entries
            orphaned = indexed_files - actual_files
            for orphaned_file in orphaned:
                # Find and remove from index
                to_remove = []
                for item_id, entry in self.index.items():
                    if entry['file_path'] == orphaned_file:
                        to_remove.append(item_id)

                for item_id in to_remove:
                    del self.index[item_id]

            await self._save_index()
            return True

        except Exception as e:
            print(f"Error optimizing storage: {e}")
            return False

    async def backup(self, backup_path: str) -> bool:
        """Create backup of storage"""
        try:
            import shutil

            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy entire storage directory
            shutil.copytree(self.base_dir, backup_dir / self.base_dir.name, dirs_exist_ok=True)
            return True

        except Exception as e:
            print(f"Error creating backup: {e}")
            return False

    async def restore(self, backup_path: str) -> bool:
        """Restore from backup"""
        try:
            import shutil

            backup_dir = Path(backup_path) / self.base_dir.name

            if not backup_dir.exists():
                return False

            # Clear current storage
            await self.clear()

            # Copy backup to storage
            shutil.copytree(backup_dir, self.base_dir, dirs_exist_ok=True)

            # Reload index
            await self.initialize()

            return True

        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False
```

#### Step 3: Implement Database Storage

```python
# storage/database_storage.py
import asyncio
import aiosqlite
import sqlite3
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
from storage.storage_interface import StorageBackend, StorageItem, DeduplicationMixin


class SQLiteStorage(StorageBackend, DeduplicationMixin):
    """SQLite database storage backend"""

    def __init__(self, db_path: str = "data/crawler_data.db", config: StorageConfig = None):
        super().__init__(config)
        self.db_path = db_path
        self.connection: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Initialize database and create tables"""
        self.connection = await aiosqlite.connect(self.db_path)

        # Create tables
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                data TEXT NOT NULL,
                crawled_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')

        # Create indexes for better performance
        await self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_url ON items(url)
        ''')

        await self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_content_hash ON items(content_hash)
        ''')

        await self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_crawled_at ON items(crawled_at)
        ''')

        await self.connection.commit()

    async def cleanup(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()

    async def store(self, item: StorageItem) -> bool:
        """Store a single item"""
        try:
            # Check for duplicates
            is_duplicate = self.is_duplicate(item.content_hash)
            if is_duplicate:
                return True

            item_dict = item.to_dict()

            await self.connection.execute('''
                INSERT OR REPLACE INTO items
                (id, url, data, crawled_at, content_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                item.id,
                item.url,
                json.dumps(item.data, ensure_ascii=False),
                item.crawled_at.isoformat(),
                item.content_hash,
                json.dumps(item.metadata or {}, ensure_ascii=False)
            ))

            await self.connection.commit()
            return True

        except Exception as e:
            print(f"Error storing item {item.id}: {e}")
            return False

    async def store_batch(self, items: List[StorageItem]) -> int:
        """Store multiple items efficiently"""
        try:
            # Use transaction for batch insert
            await self.connection.execute('BEGIN TRANSACTION')

            stored_count = 0
            for item in items:
                is_duplicate = self.is_duplicate(item.content_hash)
                if is_duplicate:
                    stored_count += 1  # Count as stored for deduplication
                    continue

                item_dict = item.to_dict()

                await self.connection.execute('''
                    INSERT OR IGNORE INTO items
                    (id, url, data, crawled_at, content_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    item.id,
                    item.url,
                    json.dumps(item.data, ensure_ascii=False),
                    item.crawled_at.isoformat(),
                    item.content_hash,
                    json.dumps(item.metadata or {}, ensure_ascii=False)
                ))

                stored_count += 1

            await self.connection.commit()
            return stored_count

        except Exception as e:
            await self.connection.rollback()
            print(f"Error in batch store: {e}")
            return 0

    async def retrieve(self, item_id: str) -> Optional[StorageItem]:
        """Retrieve a single item by ID"""
        try:
            cursor = await self.connection.execute('''
                SELECT url, data, crawled_at, content_hash, metadata
                FROM items WHERE id = ?
            ''', (item_id,))

            row = await cursor.fetchone()

            if not row:
                return None

            url, data_json, crawled_at_str, content_hash, metadata_json = row

            return StorageItem(
                id=item_id,
                url=url,
                data=json.loads(data_json),
                crawled_at=datetime.fromisoformat(crawled_at_str),
                content_hash=content_hash,
                metadata=json.loads(metadata_json) if metadata_json else {}
            )

        except Exception as e:
            print(f"Error retrieving item {item_id}: {e}")
            return None

    async def retrieve_batch(self, item_ids: List[str]) -> List[StorageItem]:
        """Retrieve multiple items by IDs"""
        if not item_ids:
            return []

        # Create placeholders for SQL query
        placeholders = ','.join('?' * len(item_ids))

        try:
            cursor = await self.connection.execute(f'''
                SELECT id, url, data, crawled_at, content_hash, metadata
                FROM items WHERE id IN ({placeholders})
            ''', item_ids)

            rows = await cursor.fetchall()
            items = []

            for row in rows:
                item_id, url, data_json, crawled_at_str, content_hash, metadata_json = row

                item = StorageItem(
                    id=item_id,
                    url=url,
                    data=json.loads(data_json),
                    crawled_at=datetime.fromisoformat(crawled_at_str),
                    content_hash=content_hash,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                items.append(item)

            return items

        except Exception as e:
            print(f"Error in batch retrieve: {e}")
            return []

    async def search(self, query: Dict[str, Any], limit: int = 100) -> List[StorageItem]:
        """Search items with query parameters"""
        try:
            where_clauses = []
            params = []

            # Build WHERE clause
            for key, value in query.items():
                if key == 'url_contains':
                    where_clauses.append('url LIKE ?')
                    params.append(f'%{value}%')
                elif key == 'content_contains':
                    where_clauses.append('data LIKE ?')
                    params.append(f'%{value}%')
                elif key == 'date_after':
                    where_clauses.append('crawled_at > ?')
                    params.append(value.isoformat())
                elif key == 'date_before':
                    where_clauses.append('crawled_at < ?')
                    params.append(value.isoformat())
                elif key == 'content_hash':
                    where_clauses.append('content_hash = ?')
                    params.append(value)

            where_sql = ' AND '.join(where_clauses) if where_clauses else '1=1'

            cursor = await self.connection.execute(f'''
                SELECT id, url, data, crawled_at, content_hash, metadata
                FROM items
                WHERE {where_sql}
                ORDER BY crawled_at DESC
                LIMIT ?
            ''', params + [limit])

            rows = await cursor.fetchall()
            items = []

            for row in rows:
                item_id, url, data_json, crawled_at_str, content_hash, metadata_json = row

                item = StorageItem(
                    id=item_id,
                    url=url,
                    data=json.loads(data_json),
                    crawled_at=datetime.fromisoformat(crawled_at_str),
                    content_hash=content_hash,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                items.append(item)

            return items

        except Exception as e:
            print(f"Error in search: {e}")
            return []

    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID"""
        try:
            cursor = await self.connection.execute('''
                DELETE FROM items WHERE id = ?
            ''', (item_id,))

            await self.connection.commit()
            return cursor.rowcount > 0

        except Exception as e:
            print(f"Error deleting item {item_id}: {e}")
            return False

    async def exists(self, item_id: str) -> bool:
        """Check if item exists"""
        try:
            cursor = await self.connection.execute('''
                SELECT 1 FROM items WHERE id = ? LIMIT 1
            ''', (item_id,))

            row = await cursor.fetchone()
            return row is not None

        except Exception as e:
            return False

    async def count(self) -> int:
        """Get total number of stored items"""
        try:
            cursor = await self.connection.execute('SELECT COUNT(*) FROM items')
            row = await cursor.fetchone()
            return row[0] if row else 0

        except Exception as e:
            return 0

    async def clear(self) -> int:
        """Clear all stored items"""
        try:
            cursor = await self.connection.execute('DELETE FROM items')
            deleted_count = cursor.rowcount
            await self.connection.commit()
            return deleted_count

        except Exception as e:
            print(f"Error clearing storage: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            # Get basic counts
            cursor = await self.connection.execute('SELECT COUNT(*) FROM items')
            total_items = (await cursor.fetchone())[0]

            # Get size information
            cursor = await self.connection.execute('''
                SELECT SUM(LENGTH(data) + LENGTH(metadata)) as total_data_size
                FROM items
            ''')
            row = await cursor.fetchone()
            total_data_size = row[0] if row and row[0] else 0

            # Get date range
            cursor = await self.connection.execute('''
                SELECT MIN(crawled_at), MAX(crawled_at) FROM items
            ''')
            row = await cursor.fetchone()
            min_date = row[0] if row else None
            max_date = row[1] if row else None

            return {
                'total_items': total_items,
                'total_data_size_bytes': total_data_size,
                'avg_item_size_bytes': total_data_size / total_items if total_items > 0 else 0,
                'date_range': {
                    'earliest': min_date,
                    'latest': max_date
                },
                'db_path': self.db_path
            }

        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    async def optimize(self) -> bool:
        """Optimize database (VACUUM and REINDEX)"""
        try:
            await self.connection.execute('VACUUM')
            await self.connection.execute('REINDEX')
            await self.connection.commit()
            return True

        except Exception as e:
            print(f"Error optimizing database: {e}")
            return False

    async def backup(self, backup_path: str) -> bool:
        """Create backup of database"""
        try:
            import shutil
            from pathlib import Path

            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            # Close current connection
            await self.cleanup()

            # Copy database file
            shutil.copy2(self.db_path, backup_path)

            # Reinitialize connection
            await self.initialize()

            return True

        except Exception as e:
            print(f"Error creating backup: {e}")
            return False

    async def restore(self, backup_path: str) -> bool:
        """Restore from backup"""
        try:
            import shutil
            from pathlib import Path

            backup_file = Path(backup_path)

            if not backup_file.exists():
                return False

            # Close current connection
            await self.cleanup()

            # Replace database file
            shutil.copy2(backup_path, self.db_path)

            # Reinitialize connection
            await self.initialize()

            return True

        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False
```

### Part 2: Data Indexing and Search

#### Step 1: Create Search Index

```python
# storage/search_index.py
import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from storage.storage_interface import StorageItem
import re


class InvertedIndex:
    """Simple inverted index for text search"""

    def __init__(self):
        self.index: Dict[str, Set[str]] = defaultdict(set)  # term -> set of item_ids
        self.item_terms: Dict[str, Set[str]] = defaultdict(set)  # item_id -> set of terms

    def add_item(self, item: StorageItem):
        """Add item to index"""
        item_id = item.id
        terms = self._extract_terms(item)

        # Remove old terms for this item
        old_terms = self.item_terms.get(item_id, set())
        for term in old_terms:
            self.index[term].discard(item_id)

        # Add new terms
        self.item_terms[item_id] = terms
        for term in terms:
            self.index[term].add(item_id)

    def remove_item(self, item_id: str):
        """Remove item from index"""
        if item_id in self.item_terms:
            terms = self.item_terms[item_id]
            for term in terms:
                self.index[term].discard(item_id)
            del self.item_terms[item_id]

    def search(self, query_terms: List[str], operator: str = 'AND') -> Set[str]:
        """Search for items containing query terms"""
        if not query_terms:
            return set()

        # Get item sets for each term
        term_sets = []
        for term in query_terms:
            item_set = self.index.get(term.lower(), set())
            term_sets.append(item_set)

        if not term_sets:
            return set()

        # Combine sets based on operator
        if operator.upper() == 'AND':
            result = term_sets[0]
            for item_set in term_sets[1:]:
                result = result.intersection(item_set)
        elif operator.upper() == 'OR':
            result = term_sets[0]
            for item_set in term_sets[1:]:
                result = result.union(item_set)
        else:
            result = set()

        return result

    def _extract_terms(self, item: StorageItem) -> Set[str]:
        """Extract searchable terms from item"""
        terms = set()

        # Extract from URL
        url_terms = self._tokenize_text(item.url)
        terms.update(url_terms)

        # Extract from data fields
        for key, value in item.data.items():
            if isinstance(value, str):
                field_terms = self._tokenize_text(value)
                terms.update(field_terms)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        field_terms = self._tokenize_text(item)
                        terms.update(field_terms)

        return terms

    def _tokenize_text(self, text: str) -> Set[str]:
        """Tokenize text into searchable terms"""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)

        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'}
        terms = {word for word in words if len(word) > 2 and word not in stop_words}

        return terms

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        total_terms = len(self.index)
        total_items = len(self.item_terms)

        term_frequencies = {}
        for term, item_ids in self.index.items():
            term_frequencies[term] = len(item_ids)

        # Most common terms
        sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_terms = dict(sorted_terms[:20])

        return {
            'total_unique_terms': total_terms,
            'total_indexed_items': total_items,
            'avg_terms_per_item': sum(len(terms) for terms in self.item_terms.values()) / total_items if total_items > 0 else 0,
            'top_terms': top_terms
        }


class SearchEngine:
    """Advanced search engine with filtering and ranking"""

    def __init__(self, index: InvertedIndex = None):
        self.index = index or InvertedIndex()
        self.item_store: Dict[str, StorageItem] = {}

    def add_item(self, item: StorageItem):
        """Add item to search engine"""
        self.index.add_item(item)
        self.item_store[item.id] = item

    def remove_item(self, item_id: str):
        """Remove item from search engine"""
        self.index.remove_item(item_id)
        self.item_store.pop(item_id, None)

    def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search with query string and filters"""
        # Parse query
        query_terms = self._parse_query(query)

        # Search index
        matching_ids = self.index.search(query_terms, operator='AND')

        # Apply filters
        filtered_results = []
        for item_id in matching_ids:
            item = self.item_store.get(item_id)
            if item and self._matches_filters(item, filters or {}):
                # Calculate relevance score
                score = self._calculate_relevance_score(item, query_terms)

                result = {
                    'item': item,
                    'score': score,
                    'matched_terms': query_terms
                }
                filtered_results.append(result)

        # Sort by score and limit
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return filtered_results[:limit]

    def _parse_query(self, query: str) -> List[str]:
        """Parse search query into terms"""
        # Simple parsing - split by spaces and handle quotes
        terms = []
        current_term = []
        in_quotes = False

        for char in query:
            if char == '"' and not in_quotes:
                in_quotes = True
            elif char == '"' and in_quotes:
                in_quotes = False
                if current_term:
                    terms.append(''.join(current_term))
                    current_term = []
            elif char.isspace() and not in_quotes:
                if current_term:
                    terms.append(''.join(current_term))
                    current_term = []
            else:
                current_term.append(char)

        if current_term:
            terms.append(''.join(current_term))

        return [term.lower() for term in terms and term]

    def _matches_filters(self, item: StorageItem, filters: Dict[str, Any]) -> bool:
        """Check if item matches filters"""
        for filter_key, filter_value in filters.items():
            if filter_key == 'date_after':
                if item.crawled_at < filter_value:
                    return False
            elif filter_key == 'date_before':
                if item.crawled_at > filter_value:
                    return False
            elif filter_key == 'url_domain':
                if filter_value not in item.url:
                    return False
            elif filter_key == 'has_field':
                if filter_value not in item.data:
                    return False
            elif filter_key.startswith('data.'):
                field_path = filter_key[5:]  # Remove 'data.' prefix
                if field_path not in item.data:
                    return False
                if item.data[field_path] != filter_value:
                    return False

        return True

    def _calculate_relevance_score(self, item: StorageItem, query_terms: List[str]) -> float:
        """Calculate relevance score for item"""
        score = 0.0

        # Term frequency scoring
        item_terms = self.index.item_terms.get(item.id, set())
        matching_terms = item_terms.intersection(set(query_terms))

        # Base score from term matches
        score += len(matching_terms) * 1.0

        # Boost for exact matches in title/important fields
        title_text = str(item.data.get('title', '')).lower()
        for term in query_terms:
            if term in title_text:
                score += 2.0

        # Boost for URL matches
        url_lower = item.url.lower()
        for term in query_terms:
            if term in url_lower:
                score += 1.5

        # Recency boost (newer items get slight boost)
        days_old = (item.crawled_at - item.crawled_at).days  # Simplified
        recency_boost = max(0, 1.0 - (days_old / 365.0))  # Boost decays over a year
        score += recency_boost * 0.5

        return score

    def get_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions for partial query"""
        partial_lower = partial_query.lower()
        suggestions = set()

        # Find terms that start with partial query
        for term in self.index.index.keys():
            if term.startswith(partial_lower) and len(term) > len(partial_lower):
                suggestions.add(term)

        # Sort by frequency
        sorted_suggestions = sorted(suggestions, key=lambda x: len(self.index.index[x]), reverse=True)

        return sorted_suggestions[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'index_stats': self.index.get_stats(),
            'total_stored_items': len(self.item_store),
            'searchable_terms': len(self.index.index)
        }
```

### Part 3: Data Backup and Recovery

#### Step 1: Create Backup System

```python
# storage/backup_manager.py
import asyncio
import json
import shutil
import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiofiles


class BackupManager:
    """Manages data backup and recovery operations"""

    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def create_backup(self, storage_backend, backup_name: Optional[str] = None,
                           include_metadata: bool = True) -> str:
        """Create a backup of the storage backend"""

        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)

        try:
            # Get storage statistics
            stats = await storage_backend.get_stats()

            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'storage_type': storage_backend.__class__.__name__,
                'stats': stats,
                'files': []
            }

            # Backup data based on storage type
            if hasattr(storage_backend, 'db_path'):  # SQLite storage
                await self._backup_sqlite(storage_backend, backup_path, manifest)
            elif hasattr(storage_backend, 'base_dir'):  # JSON file storage
                await self._backup_json_files(storage_backend, backup_path, manifest)

            # Save manifest
            manifest_file = backup_path / "manifest.json"
            async with aiofiles.open(manifest_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(manifest, ensure_ascii=False, indent=2))

            # Create compressed archive
            archive_path = await self._create_archive(backup_path, backup_name)

            # Calculate checksum
            checksum = await self._calculate_checksum(archive_path)

            # Update manifest with checksum
            manifest['archive_checksum'] = checksum
            async with aiofiles.open(manifest_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(manifest, ensure_ascii=False, indent=2))

            print(f"Backup created successfully: {archive_path}")
            return str(archive_path)

        except Exception as e:
            # Clean up failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise Exception(f"Backup failed: {e}")

    async def _backup_sqlite(self, storage, backup_path: Path, manifest: Dict[str, Any]):
        """Backup SQLite database"""
        db_backup_path = backup_path / "database.db"
        await storage.backup(str(db_backup_path))
        manifest['files'].append({
            'type': 'database',
            'path': 'database.db',
            'original_path': storage.db_path
        })

    async def _backup_json_files(self, storage, backup_path: Path, manifest: Dict[str, Any]):
        """Backup JSON file storage"""
        # Copy entire storage directory
        storage_backup_path = backup_path / "storage"
        shutil.copytree(storage.base_dir, storage_backup_path)

        # Add files to manifest
        for file_path in storage_backup_path.rglob('*'):
            if file_path.is_file():
                manifest['files'].append({
                    'type': 'data_file',
                    'path': str(file_path.relative_to(backup_path)),
                    'original_path': str(file_path.relative_to(storage_backup_path))
                })

    async def _create_archive(self, backup_path: Path, backup_name: str) -> Path:
        """Create compressed archive of backup"""
        import tarfile

        archive_path = self.backup_dir / f"{backup_name}.tar.gz"

        def create_archive():
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_name)

        # Run in thread pool since tarfile is synchronous
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, create_archive)

        # Remove uncompressed backup directory
        shutil.rmtree(backup_path)

        return archive_path

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()

        def update_hash():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_hash)

        return sha256.hexdigest()

    async def restore_backup(self, archive_path: str, storage_backend,
                           verify_checksum: bool = True) -> bool:
        """Restore from backup archive"""

        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")

        # Verify checksum if requested
        if verify_checksum:
            await self._verify_backup_checksum(archive_path)

        # Extract archive
        extract_path = self.backup_dir / "temp_restore"
        extract_path.mkdir(exist_ok=True)

        try:
            import tarfile

            def extract_archive():
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(extract_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, extract_archive)

            # Find backup directory
            backup_dirs = list(extract_path.glob("*"))
            if not backup_dirs:
                raise Exception("No backup directory found in archive")

            backup_path = backup_dirs[0]

            # Load manifest
            manifest_file = backup_path / "manifest.json"
            async with aiofiles.open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.loads(await f.read())

            # Restore based on storage type
            storage_type = manifest.get('storage_type', '')

            if 'SQLite' in storage_type:
                success = await self._restore_sqlite(backup_path, storage_backend, manifest)
            elif 'JSONFile' in storage_type:
                success = await self._restore_json_files(backup_path, storage_backend, manifest)
            else:
                raise Exception(f"Unknown storage type: {storage_type}")

            return success

        finally:
            # Clean up temp directory
            if extract_path.exists():
                shutil.rmtree(extract_path)

    async def _verify_backup_checksum(self, archive_path: Path):
        """Verify backup archive checksum"""
        # Extract manifest from archive to check checksum
        import tarfile

        with tarfile.open(archive_path, 'r:gz') as tar:
            # Find manifest file
            manifest_member = None
            for member in tar.getmembers():
                if member.name.endswith('/manifest.json'):
                    manifest_member = member
                    break

            if not manifest_member:
                raise Exception("Manifest not found in backup archive")

            # Extract manifest
            manifest_content = tar.extractfile(manifest_member).read().decode('utf-8')
            manifest = json.loads(manifest_content)

            expected_checksum = manifest.get('archive_checksum')
            if not expected_checksum:
                print("Warning: No checksum found in manifest, skipping verification")
                return

            # Calculate actual checksum
            actual_checksum = await self._calculate_checksum(archive_path)

            if actual_checksum != expected_checksum:
                raise Exception(f"Checksum verification failed. Expected: {expected_checksum}, Got: {actual_checksum}")

    async def _restore_sqlite(self, backup_path: Path, storage_backend, manifest: Dict[str, Any]) -> bool:
        """Restore SQLite database"""
        db_file = backup_path / "database.db"
        if not db_file.exists():
            raise Exception("Database file not found in backup")

        return await storage_backend.restore(str(db_file))

    async def _restore_json_files(self, backup_path: Path, storage_backend, manifest: Dict[str, Any]) -> bool:
        """Restore JSON file storage"""
        storage_dir = backup_path / "storage"
        if not storage_dir.exists():
            raise Exception("Storage directory not found in backup")

        # Clear current storage
        await storage_backend.clear()

        # Copy files back
        for file_info in manifest.get('files', []):
            if file_info['type'] == 'data_file':
                src_path = backup_path / file_info['path']
                dst_path = storage_backend.base_dir / file_info['original_path']

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

        # Reinitialize to reload index
        await storage_backend.initialize()
        return True

    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []

        for archive_path in self.backup_dir.glob("*.tar.gz"):
            try:
                # Extract manifest info without full extraction
                import tarfile

                with tarfile.open(archive_path, 'r:gz') as tar:
                    manifest_member = None
                    for member in tar.getmembers():
                        if member.name.endswith('/manifest.json'):
                            manifest_member = member
                            break

                    if manifest_member:
                        manifest_content = tar.extractfile(manifest_member).read().decode('utf-8')
                        manifest = json.loads(manifest_content)

                        backup_info = {
                            'name': manifest.get('backup_name', archive_path.stem),
                            'created_at': manifest.get('created_at'),
                            'archive_path': str(archive_path),
                            'stats': manifest.get('stats', {}),
                            'checksum': manifest.get('archive_checksum')
                        }
                        backups.append(backup_info)

            except Exception as e:
                print(f"Error reading backup {archive_path}: {e}")

        # Sort by creation date
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return backups

    async def cleanup_old_backups(self, keep_last: int = 5):
        """Clean up old backups, keeping the most recent ones"""
        backups = await self.list_backups()

        if len(backups) <= keep_last:
            return 0

        # Remove old backups
        removed_count = 0
        for backup in backups[keep_last:]:
            archive_path = Path(backup['archive_path'])
            try:
                archive_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Error removing backup {archive_path}: {e}")

        return removed_count
```

## Exercises

### Exercise 1: Storage Backend Implementation
1. Implement a Redis storage backend for high-performance caching
2. Create a cloud storage backend using AWS S3 or Google Cloud Storage
3. Add encryption support to existing storage backends
4. Implement storage migration between different backends

### Exercise 2: Advanced Search and Indexing
1. Add full-text search capabilities with relevance ranking
2. Implement faceted search with filters and aggregations
3. Create search analytics and query performance monitoring
4. Add support for different languages and text analysis

### Exercise 3: Backup and Recovery
1. Implement incremental backups to reduce storage and transfer time
2. Create automated backup scheduling with retention policies
3. Add backup encryption and secure key management
4. Implement cross-region backup replication

### Exercise 4: Data Integrity and Monitoring
1. Add data validation and integrity checks during storage and retrieval
2. Implement comprehensive monitoring and alerting for storage operations
3. Create data quality metrics and reporting
4. Add automatic data repair and consistency checking

## Next Steps
- Complete [Workshop 06: Deployment and Scaling](../workshops/workshop-06-deployment-scaling.md)
- Learn about production deployment strategies
- Explore monitoring and maintenance best practices

## Resources
- [Database Design and Optimization](https://example.com/database-design)
- [Search Engine Implementation](https://example.com/search-engines)
- [Data Backup Strategies](https://example.com/backup-strategies)
- [Storage Performance Tuning](https://example.com/storage-performance)