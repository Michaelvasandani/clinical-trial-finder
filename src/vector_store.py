"""Vector storage and retrieval using PostgreSQL or FAISS."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_MAPPING_PATH,
    SIMILARITY_METRIC,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE,
    VECTOR_STORAGE_TYPE,
    DATABASE_CONFIG,
    DB_CONNECTION_POOL_SIZE,
    DB_MAX_OVERFLOW,
    DB_BATCH_INSERT_SIZE,
    VECTOR_SIMILARITY_METHOD
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Store and retrieve embeddings using PostgreSQL or FAISS."""

    def __init__(
        self,
        embedding_dim: int = 768,
        similarity_metric: str = SIMILARITY_METRIC,
        storage_type: str = VECTOR_STORAGE_TYPE
    ):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings
            similarity_metric: Similarity metric to use
            storage_type: Storage backend ('postgresql' or 'faiss')
        """
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.storage_type = storage_type

        # FAISS-specific attributes
        self.index = None
        self.metadata = []
        self.chunk_mapping = {}

        # PostgreSQL-specific attributes
        self.connection_pool = None

        if self.storage_type == "postgresql":
            self._init_postgresql()

    def _init_postgresql(self):
        """Initialize PostgreSQL connection pool."""
        try:
            self.connection_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=DB_CONNECTION_POOL_SIZE,
                **DATABASE_CONFIG
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise

    def _get_connection(self):
        """Get a connection from the pool."""
        if self.connection_pool is None:
            raise RuntimeError("PostgreSQL connection pool not initialized")
        return self.connection_pool.getconn()

    def _return_connection(self, conn, close=False):
        """Return a connection to the pool."""
        if self.connection_pool is None:
            return
        if close:
            self.connection_pool.putconn(conn, close=True)
        else:
            self.connection_pool.putconn(conn)

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding array to bytes."""
        buffer = io.BytesIO()
        np.save(buffer, embedding)
        return buffer.getvalue()

    def _deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize embedding bytes to numpy array."""
        buffer = io.BytesIO(embedding_bytes)
        return np.load(buffer)

    def add_embeddings_postgresql(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        chunk_mapping: Dict[int, str]
    ):
        """Add embeddings to PostgreSQL database."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Prepare batch insert data
                insert_data = []
                for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                    nct_id = meta.get("NCTId", "")
                    chunk_id = meta.get("chunk_id", f"chunk_{i}")

                    # Serialize embedding
                    embedding_bytes = self._serialize_embedding(embedding)

                    insert_data.append((
                        nct_id,
                        chunk_id,
                        embedding_bytes,
                        json.dumps(meta)
                    ))

                # Batch insert in chunks
                batch_size = DB_BATCH_INSERT_SIZE
                for i in range(0, len(insert_data), batch_size):
                    batch = insert_data[i:i + batch_size]

                    psycopg2.extras.execute_batch(
                        cursor,
                        """
                        INSERT INTO trial_embeddings (nct_id, chunk_id, embedding_data, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (nct_id, chunk_id) DO UPDATE SET
                            embedding_data = EXCLUDED.embedding_data,
                            metadata = EXCLUDED.metadata
                        """,
                        batch
                    )

                conn.commit()
                logger.info(f"Inserted {len(insert_data)} embeddings into PostgreSQL")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting embeddings: {e}")
            raise
        finally:
            self._return_connection(conn)

    def search_postgresql(
        self,
        query_embedding: np.ndarray,
        k: int = DEFAULT_TOP_K,
        filter_metadata: Optional[Dict] = None,
        min_score: float = MIN_SIMILARITY_SCORE
    ) -> List[Dict]:
        """Search for similar documents in PostgreSQL."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Get all embeddings and compute similarity in Python
                # This is simpler than complex SQL and works reliably
                cursor.execute("""
                    SELECT id, nct_id, chunk_id, embedding_data, metadata
                    FROM trial_embeddings
                    ORDER BY id
                """)

                results = []
                seen_nct_ids = set()

                # Normalize query embedding for cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    query_embedding = query_embedding / query_norm

                for row in cursor:
                    # Deserialize embedding
                    stored_embedding = self._deserialize_embedding(row['embedding_data'])

                    # Normalize stored embedding
                    stored_norm = np.linalg.norm(stored_embedding)
                    if stored_norm > 0:
                        stored_embedding = stored_embedding / stored_norm

                    # Compute cosine similarity
                    similarity = float(np.dot(query_embedding, stored_embedding))

                    # Apply minimum score filter
                    if similarity < min_score:
                        continue

                    # Get metadata (already parsed as dict from JSONB)
                    meta = row['metadata']
                    nct_id = row['nct_id']

                    # Skip if we've already seen this trial
                    if nct_id in seen_nct_ids:
                        continue

                    # Apply metadata filters
                    if filter_metadata:
                        skip = False
                        for key, value in filter_metadata.items():
                            if key in meta and meta[key] != value:
                                skip = True
                                break
                        if skip:
                            continue

                    results.append({
                        "NCTId": nct_id,
                        "score": similarity,
                        "metadata": meta,
                        "chunk_index": meta.get("chunk_index", 0),
                        "total_chunks": meta.get("total_chunks", 1)
                    })

                    seen_nct_ids.add(nct_id)

                # Sort by similarity score (descending) and return top k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:k]

        except Exception as e:
            logger.error(f"Error searching PostgreSQL: {e}")
            raise
        finally:
            self._return_connection(conn)
        
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
        if self.storage_type == "postgresql":
            self.add_embeddings_postgresql(embeddings, metadata, chunk_mapping)
        else:
            # FAISS implementation
            if self.index is None:
                self.create_index(embeddings)
            else:
                # Add to existing index
                embeddings = embeddings.astype(np.float32)
                if self.similarity_metric == "cosine":
                    import faiss
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
        if self.storage_type == "postgresql":
            return self.search_postgresql(query_embedding, k, filter_metadata, min_score)
        else:
            # FAISS implementation
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty")
                return []

            # Prepare query
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

            if self.similarity_metric == "cosine":
                import faiss
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
        if self.storage_type == "postgresql":
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    # Get total count
                    cursor.execute("SELECT COUNT(*) FROM trial_embeddings")
                    total_vectors = cursor.fetchone()[0]

                    # Get unique trials
                    cursor.execute("SELECT COUNT(DISTINCT nct_id) FROM trial_embeddings")
                    unique_trials = cursor.fetchone()[0]

                    # Get status distribution
                    cursor.execute("""
                        SELECT
                            metadata->>'OverallStatus' as status,
                            COUNT(*) as count
                        FROM trial_embeddings
                        GROUP BY metadata->>'OverallStatus'
                    """)
                    status_distribution = {row[0] or "Unknown": row[1] for row in cursor.fetchall()}

                    return {
                        "total_vectors": total_vectors,
                        "embedding_dim": self.embedding_dim,
                        "similarity_metric": self.similarity_metric,
                        "storage_type": "postgresql",
                        "unique_trials": unique_trials,
                        "status_distribution": status_distribution
                    }
            finally:
                self._return_connection(conn)
        else:
            # FAISS implementation
            stats = {
                "total_vectors": self.index.ntotal if self.index else 0,
                "embedding_dim": self.embedding_dim,
                "similarity_metric": self.similarity_metric,
                "storage_type": "faiss",
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

    def save(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        chunk_mapping_path: Optional[Path] = None
    ):
        """
        Save index and metadata to disk.

        Args:
            index_path: Path to save FAISS index (ignored for PostgreSQL)
            metadata_path: Path to save metadata (ignored for PostgreSQL)
            chunk_mapping_path: Path to save chunk mapping (ignored for PostgreSQL)
        """
        if self.storage_type == "postgresql":
            logger.info("PostgreSQL data is automatically persisted")
            return

        # FAISS implementation
        index_path = index_path or FAISS_INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH
        chunk_mapping_path = chunk_mapping_path or CHUNK_MAPPING_PATH

        # Save FAISS index
        if self.index is not None:
            logger.info(f"Saving FAISS index to {index_path}")
            import faiss
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
            index_path: Path to FAISS index (ignored for PostgreSQL)
            metadata_path: Path to metadata (ignored for PostgreSQL)
            chunk_mapping_path: Path to chunk mapping (ignored for PostgreSQL)
        """
        if self.storage_type == "postgresql":
            logger.info("PostgreSQL data is loaded on-demand from database")
            return

        # FAISS implementation
        index_path = index_path or FAISS_INDEX_PATH
        metadata_path = metadata_path or METADATA_PATH
        chunk_mapping_path = chunk_mapping_path or CHUNK_MAPPING_PATH

        # Load FAISS index
        if index_path.exists():
            logger.info(f"Loading FAISS index from {index_path}")
            import faiss
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