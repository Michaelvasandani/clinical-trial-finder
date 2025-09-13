"""Vector storage and retrieval using FAISS."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from config_embeddings import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_MAPPING_PATH,
    SIMILARITY_METRIC,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Store and retrieve embeddings using FAISS."""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        similarity_metric: str = SIMILARITY_METRIC
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            similarity_metric: Similarity metric to use
        """
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.index = None
        self.metadata = []
        self.chunk_mapping = {}
        
    def create_index(self, embeddings: np.ndarray):
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings (num_docs, embedding_dim)
        """
        logger.info(f"Creating FAISS index for {len(embeddings)} embeddings...")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Create appropriate index based on similarity metric
        if self.similarity_metric == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.similarity_metric == "inner_product":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.similarity_metric == "euclidean":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        logger.info(f"Index created with {self.index.ntotal} vectors")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        chunk_mapping: Dict[int, str]
    ):
        """
        Add embeddings with metadata to the store.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of metadata dictionaries
            chunk_mapping: Mapping of chunk indices to NCT IDs
        """
        if self.index is None:
            self.create_index(embeddings)
        else:
            # Add to existing index
            embeddings = embeddings.astype(np.float32)
            if self.similarity_metric == "cosine":
                faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        # Store metadata and mapping
        self.metadata.extend(metadata)
        
        # Update chunk mapping with new indices
        current_size = len(self.chunk_mapping)
        for local_idx, nct_id in chunk_mapping.items():
            self.chunk_mapping[current_size + local_idx] = nct_id
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = DEFAULT_TOP_K,
        filter_metadata: Optional[Dict] = None,
        min_score: float = MIN_SIMILARITY_SCORE
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            min_score: Minimum similarity score
            
        Returns:
            List of search results with metadata and scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Prepare query
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k * 2, self.index.ntotal))
        
        # Process results
        results = []
        seen_nct_ids = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            
            # Get metadata
            doc_metadata = self.metadata[idx]
            nct_id = doc_metadata.get("NCTId", "")
            
            # Skip if we've already seen this trial (different chunks)
            if nct_id in seen_nct_ids:
                continue
            
            # Apply metadata filters if provided
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key in doc_metadata and doc_metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Apply minimum score filter
            if self.similarity_metric == "euclidean":
                # Convert distance to similarity score
                similarity = 1 / (1 + score)
            else:
                similarity = float(score)
            
            if similarity < min_score:
                continue
            
            # Add to results
            results.append({
                "NCTId": nct_id,
                "score": similarity,
                "metadata": doc_metadata,
                "chunk_index": doc_metadata.get("chunk_index", 0),
                "total_chunks": doc_metadata.get("total_chunks", 1)
            })
            
            seen_nct_ids.add(nct_id)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_nct_id(self, nct_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific NCT ID.
        
        Args:
            nct_id: NCT identifier
            
        Returns:
            Metadata dictionary or None
        """
        for metadata in self.metadata:
            if metadata.get("NCTId") == nct_id:
                return metadata
        return None
    
    def save(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        chunk_mapping_path: Optional[Path] = None
    ):
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
            chunk_mapping_path: Path to save chunk mapping
        """
        index_path = index_path or FAISS_INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH
        chunk_mapping_path = chunk_mapping_path or CHUNK_MAPPING_PATH
        
        # Save FAISS index
        if self.index is not None:
            logger.info(f"Saving FAISS index to {index_path}")
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save chunk mapping
        logger.info(f"Saving chunk mapping to {chunk_mapping_path}")
        with open(chunk_mapping_path, 'w') as f:
            json.dump(self.chunk_mapping, f, indent=2)
    
    def load(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        chunk_mapping_path: Optional[Path] = None
    ):
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
            chunk_mapping_path: Path to chunk mapping
        """
        index_path = index_path or FAISS_INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH
        chunk_mapping_path = chunk_mapping_path or CHUNK_MAPPING_PATH
        
        # Load FAISS index
        if index_path.exists():
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        
        # Load metadata
        if metadata_path.exists():
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")
        
        # Load chunk mapping
        if chunk_mapping_path.exists():
            logger.info(f"Loading chunk mapping from {chunk_mapping_path}")
            with open(chunk_mapping_path, 'r') as f:
                self.chunk_mapping = json.load(f)
            logger.info(f"Loaded mapping for {len(self.chunk_mapping)} chunks")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "similarity_metric": self.similarity_metric,
            "metadata_entries": len(self.metadata),
            "unique_trials": len(set(m.get("NCTId", "") for m in self.metadata))
        }
        
        # Get status distribution
        if self.metadata:
            statuses = {}
            for m in self.metadata:
                status = m.get("OverallStatus", "Unknown")
                statuses[status] = statuses.get(status, 0) + 1
            stats["status_distribution"] = statuses
        
        return stats