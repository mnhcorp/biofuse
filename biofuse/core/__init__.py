"""
Core BioFuse functionality including main API and caching.
"""

from .cache import EmbeddingCache, get_cache

__all__ = ['EmbeddingCache', 'get_cache']
