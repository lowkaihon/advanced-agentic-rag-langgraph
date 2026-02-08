"""Redis-backed semantic cache using embedding similarity.

Caches full pipeline responses keyed by query embedding similarity.
Novel queries run the full RAG pipeline; similar queries return cached answers.
Graceful degradation: system works identically when Redis is unavailable.
"""

import json
import os
import time
import uuid
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCache:
    """Redis-backed semantic cache using embedding similarity.

    Caches full pipeline responses keyed by query embedding similarity.
    Novel queries run the full RAG pipeline; similar queries return cached answers.
    Graceful degradation: system works identically when Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        corpus_version: Optional[str] = None,
        enabled: Optional[bool] = None,
        redis_client=None,
    ):
        self._enabled = enabled if enabled is not None else (
            os.getenv("CACHE_ENABLED", "false").lower() == "true"
        )
        self._similarity_threshold = similarity_threshold or float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")
        )
        self._corpus_version = corpus_version or os.getenv("CORPUS_VERSION", "v1")
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Stats
        self._hits = 0
        self._misses = 0

        # Redis connection (graceful fail)
        self._redis = None
        if self._enabled:
            self._connect(redis_client)

        # Embedding model (lazy init)
        self._embeddings = None

    def _connect(self, redis_client=None):
        """Connect to Redis, gracefully degrading on failure."""
        if redis_client is not None:
            self._redis = redis_client
            return

        try:
            import redis
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("Semantic cache connected to Redis at %s", self._redis_url)
        except Exception as e:
            logger.warning("Semantic cache: Redis unavailable (%s), caching disabled", e)
            self._redis = None

    def _get_embeddings(self):
        """Lazy-init embedding model."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return self._embeddings

    def _key_prefix(self) -> str:
        return f"semcache:{self._corpus_version}"

    def _entry_key(self, entry_id: str) -> str:
        return f"{self._key_prefix()}:entry:{entry_id}"

    def _index_key(self) -> str:
        return f"{self._key_prefix()}:index"

    @property
    def available(self) -> bool:
        """Whether cache is enabled and Redis is connected."""
        return self._enabled and self._redis is not None

    def lookup(self, query: str) -> Optional[dict]:
        """Look up a semantically similar cached response.

        Args:
            query: The user query to look up.

        Returns:
            Dict with {response, similarity} if cache hit, None if miss.
        """
        if not self.available:
            self._misses += 1
            return None

        try:
            query_embedding = np.array(
                self._get_embeddings().embed_query(query), dtype=np.float32
            )

            # Get all entry keys from the index
            entry_keys = self._redis.smembers(self._index_key())
            if not entry_keys:
                self._misses += 1
                return None

            best_similarity = -1.0
            best_response = None

            for entry_key in entry_keys:
                entry_data = self._redis.hgetall(entry_key)
                if not entry_data or "embedding" not in entry_data:
                    continue

                cached_embedding = np.array(
                    json.loads(entry_data["embedding"]), dtype=np.float32
                )
                similarity = _cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_response = entry_data

            if best_similarity >= self._similarity_threshold and best_response:
                self._hits += 1
                response = json.loads(best_response["response"])
                return {
                    "response": response,
                    "similarity": round(best_similarity, 4),
                }

            self._misses += 1
            return None

        except Exception as e:
            logger.warning("Semantic cache lookup failed: %s", e)
            self._misses += 1
            return None

    def store(self, query: str, response: dict) -> None:
        """Store a query-response pair in the cache.

        Args:
            query: The original user query.
            response: Dict of response fields to cache.
        """
        if not self.available:
            return

        try:
            query_embedding = self._get_embeddings().embed_query(query)

            entry_id = str(uuid.uuid4())
            entry_key = self._entry_key(entry_id)

            self._redis.hset(entry_key, mapping={
                "embedding": json.dumps(query_embedding),
                "response": json.dumps(response),
                "query_text": query,
                "timestamp": str(time.time()),
            })

            # Add to index set
            self._redis.sadd(self._index_key(), entry_key)

            logger.info("Semantic cache: stored entry for query '%s'", query[:50])

        except Exception as e:
            logger.warning("Semantic cache store failed: %s", e)

    def flush(self) -> int:
        """Delete all cache entries for current corpus version.

        Returns:
            Number of entries deleted.
        """
        if not self.available:
            return 0

        try:
            entry_keys = self._redis.smembers(self._index_key())
            count = 0
            for entry_key in entry_keys:
                self._redis.delete(entry_key)
                count += 1
            self._redis.delete(self._index_key())

            # Reset stats
            self._hits = 0
            self._misses = 0

            logger.info("Semantic cache: flushed %d entries", count)
            return count

        except Exception as e:
            logger.warning("Semantic cache flush failed: %s", e)
            return 0

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with hits, misses, total_queries, hit_rate, cache_size.
        """
        total = self._hits + self._misses
        cache_size = 0

        if self.available:
            try:
                cache_size = self._redis.scard(self._index_key())
            except Exception:
                pass

        return {
            "enabled": self._enabled,
            "available": self.available,
            "hits": self._hits,
            "misses": self._misses,
            "total_queries": total,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "cache_size": cache_size,
            "similarity_threshold": self._similarity_threshold,
            "corpus_version": self._corpus_version,
        }
