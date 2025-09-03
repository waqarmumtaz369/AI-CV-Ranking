"""
Singleton module for embedding models to prevent multiple loads.
"""
import os
import threading
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Global cache with thread safety
_EMBEDDING_MODEL_CACHE = {}
_cache_lock = threading.Lock()

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """Get or create embedding model singleton with thread safety."""
    with _cache_lock:
        if model_name not in _EMBEDDING_MODEL_CACHE:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    print(f"🔄 Loading embedding model: {model_name}")
                    _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
                    print(f"✅ Successfully loaded embedding model: {model_name}")
                except Exception as e:
                    print(f"❌ Error loading embedding model {model_name}: {e}")
                    _EMBEDDING_MODEL_CACHE[model_name] = None
            else:
                print("⚠️ sentence-transformers not available")
                _EMBEDDING_MODEL_CACHE[model_name] = None
        else:
            print(f"♻️ Using cached embedding model: {model_name}")
    
    return _EMBEDDING_MODEL_CACHE[model_name]

def clear_embedding_cache():
    """Clear the embedding model cache."""
    with _cache_lock:
        _EMBEDDING_MODEL_CACHE.clear()
        print("🗑️ Cleared embedding model cache")

def get_cache_info():
    """Get information about the cache."""
    with _cache_lock:
        return {
            "cached_models": list(_EMBEDDING_MODEL_CACHE.keys()),
            "cache_size": len(_EMBEDDING_MODEL_CACHE)
        }
