import os
import pickle
from functools import lru_cache
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.utils import logger


class HierarchicalSemanticMatcher:
    """
    Complete semantic matching solution with embedding truncation,
    FAISS indexing, and hierarchical search capabilities.
    """

    MIN_HIERARCHICAL_ITEMS = 1000  # Minimum items for hierarchical index

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None,  # None = use full embeddings
        use_hierarchical: bool = True,
        n_clusters: int = 100,
        cache_dir: str = "embeddings_cache",
        batch_size: int = 500,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.use_hierarchical = use_hierarchical
        self.n_clusters = n_clusters
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        # Initialize model
        self.model = SentenceTransformer(model_name)
        self.original_dim = self.model.get_sentence_embedding_dimension()

        # Set final embedding dimension
        if embedding_dim and embedding_dim < self.original_dim:
            self.final_dim = embedding_dim
            logger.info(f"Enable embeddings compression from {self.original_dim} to {self.final_dim}")
        else:
            self.final_dim = self.original_dim

        # Storage
        self.items = None
        self.embeddings = None
        self.faiss_index = None
        self.hierarchical_index = None

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _compress_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress embeddings by truncating to specified dimensions."""
        if self.embedding_dim and self.embedding_dim < embeddings.shape[1]:
            # Simple truncation - surprisingly effective!
            return embeddings[:, : self.embedding_dim]
        return embeddings

    def _get_cache_path(self, data_identifier: str) -> str:
        """Get cache file path for embeddings."""
        cache_name = f"{self.model_name.replace('/', '_')}_{data_identifier}_{self.final_dim}d.pkl"
        return os.path.join(self.cache_dir, cache_name)

    def encode_items(self, items: list[str], data_identifier: str = "semantic_data", force_recompute: bool = False):
        """
        Encode items with caching and compression.

        Args:
            items: List of item names
            data_identifier: Unique identifier for caching
            force_recompute: Whether to recompute even if cache exists
        """
        cache_path = self._get_cache_path(data_identifier)

        # Try to load from cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading embeddings from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.items = cached_data["items"]
                    self.embeddings = cached_data["embeddings"]

                if len(self.items) == len(items):
                    logger.info("Cache loaded successfully")
                    self._build_indices()
                    return
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(items)} items...")
        self.items = items

        all_embeddings = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(items) - 1) // self.batch_size + 1}")

            batch_embeddings = self.model.encode(batch, show_progress_bar=True, convert_to_numpy=True)

            # Compress embeddings
            batch_embeddings = self._compress_embeddings(batch_embeddings)
            all_embeddings.append(batch_embeddings)

        self.embeddings = np.vstack(all_embeddings)

        # Cache the results
        cache_data = {"items": self.items, "embeddings": self.embeddings}

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"Embeddings cached to: {cache_path}")
        except Exception as e:
            logger.warning(f"Caching failed: {e}")

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build FAISS indices for fast similarity search."""
        logger.info("Building FAISS indices...")

        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)

        if self.use_hierarchical and len(self.items) > HierarchicalSemanticMatcher.MIN_HIERARCHICAL_ITEMS:
            # Hierarchical index for large datasets
            quantizer = faiss.IndexFlatIP(self.final_dim)
            nlist = min(self.n_clusters, len(self.items) // 10)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.final_dim, nlist)

            # Train the index
            logger.info("Training hierarchical index...")
            self.faiss_index.train(embeddings_normalized)
            self.faiss_index.add(embeddings_normalized)

            # Set search parameters
            self.faiss_index.nprobe = min(max(16, nlist // 8), 128)
        else:
            # Flat index for smaller datasets
            self.faiss_index = faiss.IndexFlatIP(self.final_dim)
            self.faiss_index.add(embeddings_normalized)

        logger.info(f"Index built with {self.faiss_index.ntotal} items")
        logger.info(self.get_stats())

    @lru_cache(maxsize=128)
    def find_similar_items(self, query: str, top_k: int = 10, max_distance: float = 0.8) -> list[tuple[str, float]]:
        """
        Find similar items using semantic search.

        Args:
            query: Item to search for
            top_k: Number of results to return
            max_distance: Maximum distance threshold

        Returns:
            List of (item, cosine_distance) tuples
        """
        logger.info(f"Finding similar items for query: '{query}' with top_k={top_k} and max_distance={max_distance}")
        if self.faiss_index is None:
            raise ValueError("Index not built. Call encode_items() first.")

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self._compress_embeddings(query_embedding)
        logger.info(f"Query compressed embedding shape: {query_embedding.shape}")

        # Normalize for cosine distance
        faiss.normalize_L2(query_embedding)
        logger.info("Query embedding normalized")

        # Search
        distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.items)))
        logger.info(f"Search completed. Found {len(distances[0])} candidates with cosine distances {distances[0]}")
        logger.info(f"Candidates: {self.items[indices[0]]}")

        # Filter and format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist <= max_distance:  # -1 indicates no result
                item = self.items[idx]
                results.append((item, float(dist)))
        logger.info(f"Filtered results: {len(results)} items with cosine distance <= {max_distance}")

        return results[:top_k]

    def find_most_similar_item(self, query: str, max_distance: float = 0.8) -> Optional[tuple[str, float]]:
        """
        Find the most similar item to the query.

        Args:
            query: Item to search for
            max_distance: Maximum distance threshold

        Returns:
            Tuple of (item, cosine_distance) or None if no match found
        """
        results = self.find_similar_items(query, top_k=1, max_distance=max_distance)
        return results[0] if results else None

    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        if self.embeddings is None:
            return {"status": "not_initialized"}

        memory_mb = self.embeddings.nbytes / (1024 * 1024)
        compression_ratio = self.original_dim / self.final_dim if self.final_dim < self.original_dim else 1.0

        return {
            "num_items": len(self.items),
            "embedding_dim": self.final_dim,
            "original_dim": self.original_dim,
            "compression_ratio": f"{compression_ratio:.1f}x",
            "memory_usage_mb": f"{memory_mb:.1f}",
            "index_type": (
                "hierarchical"
                if self.use_hierarchical and len(self.items) > HierarchicalSemanticMatcher.MIN_HIERARCHICAL_ITEMS
                else "flat"
            ),
        }
