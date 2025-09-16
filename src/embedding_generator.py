"""Generate embeddings using BioBERT model."""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME as MODEL_NAME,
    BATCH_SIZE,
    USE_GPU,
    SHOW_PROGRESS,
    EMBEDDING_DIM
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for clinical trial documents using BioBERT."""
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        use_gpu: bool = USE_GPU,
        batch_size: int = BATCH_SIZE
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            use_gpu: Whether to use GPU if available
            batch_size: Batch size for encoding
        """
        self.batch_size = batch_size
        
        # Check for GPU availability
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif use_gpu and torch.backends.mps.is_available():
            self.device = 'mps'
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            self.device = 'cpu'
            logger.info("Using CPU for embeddings")
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Verify embedding dimension
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        self.embedding_dim = len(test_embedding)
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        if self.embedding_dim != EMBEDDING_DIM:
            logger.warning(
                f"Embedding dimension {self.embedding_dim} doesn't match "
                f"expected {EMBEDDING_DIM}"
            )
    
    def generate_embeddings(
        self,
        documents: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = SHOW_PROGRESS,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document strings
            batch_size: Override default batch size
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            Array of embeddings (num_documents, embedding_dim)
        """
        if not documents:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        # Encode documents
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def generate_query_embedding(
        self,
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            normalize: Whether to normalize embedding
            
        Returns:
            Query embedding vector
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embedding
    
    def generate_embeddings_batched(
        self,
        documents: List[str],
        save_callback: Optional[callable] = None,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings in batches with optional saving.
        
        Args:
            documents: List of document strings
            save_callback: Function to call after each batch
            batch_size: Override default batch size
            
        Returns:
            Array of all embeddings
        """
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        with tqdm(total=len(documents), desc="Generating embeddings") as pbar:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_docs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Call save callback if provided
                if save_callback:
                    save_callback(batch_embeddings, i)
                
                pbar.update(len(batch_docs))
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        return embeddings
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document embedding matrix
            metric: Similarity metric (cosine, euclidean, inner_product)
            
        Returns:
            Similarity scores
        """
        if metric == "cosine":
            # Assuming embeddings are normalized
            similarities = np.dot(document_embeddings, query_embedding)
        elif metric == "inner_product":
            similarities = np.dot(document_embeddings, query_embedding)
        elif metric == "euclidean":
            # Convert to similarity (inverse of distance)
            distances = np.linalg.norm(
                document_embeddings - query_embedding, 
                axis=1
            )
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return similarities
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": MODEL_NAME,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "batch_size": self.batch_size
        }