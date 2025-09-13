"""Main script to generate embeddings for all clinical trials."""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config_embeddings import (
    INPUT_CSV,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_MAPPING_PATH,
    BATCH_SIZE,
    USE_GPU
)
from embeddings import (
    DocumentProcessor,
    EmbeddingGenerator,
    VectorStore
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_embeddings_for_trials(
    csv_path: Path = INPUT_CSV,
    use_chunking: bool = True,
    test_mode: bool = False,
    test_samples: int = 10
):
    """
    Generate embeddings for all clinical trials.
    
    Args:
        csv_path: Path to CSV file with trial data
        use_chunking: Whether to chunk long documents
        test_mode: If True, only process a few samples
        test_samples: Number of samples for test mode
    """
    logger.info("=" * 60)
    logger.info("Starting Clinical Trials Embedding Generation")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} trials")
    
    if test_mode:
        logger.info(f"TEST MODE: Processing only {test_samples} samples")
        df = df.head(test_samples)
    
    # Initialize components
    logger.info("Initializing components...")
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator(use_gpu=USE_GPU)
    vector_store = VectorStore(
        embedding_dim=embedding_generator.embedding_dim
    )
    
    # Process documents
    logger.info("Processing documents...")
    start_time = time.time()
    
    documents, metadata, chunk_mapping = doc_processor.process_trials(
        df, 
        chunk=use_chunking
    )
    
    process_time = time.time() - start_time
    logger.info(f"Document processing completed in {process_time:.2f} seconds")
    logger.info(f"Created {len(documents)} document chunks")
    
    # Generate embeddings
    logger.info("Generating embeddings with BioBERT...")
    start_time = time.time()
    
    embeddings = embedding_generator.generate_embeddings(
        documents,
        batch_size=BATCH_SIZE,
        show_progress=True,
        normalize=True
    )
    
    embed_time = time.time() - start_time
    logger.info(f"Embedding generation completed in {embed_time:.2f} seconds")
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create vector store
    logger.info("Creating vector store...")
    vector_store.add_embeddings(
        embeddings,
        metadata,
        chunk_mapping
    )
    
    # Save to disk
    logger.info("Saving embeddings and metadata...")
    vector_store.save()
    
    # Print statistics
    stats = vector_store.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total vectors: {stats['total_vectors']}")
    logger.info(f"Unique trials: {stats['unique_trials']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Document processing time: {process_time:.2f} seconds")
    logger.info(f"Embedding generation time: {embed_time:.2f} seconds")
    logger.info(f"Total time: {process_time + embed_time:.2f} seconds")
    
    if 'status_distribution' in stats:
        logger.info("\nTrial status distribution:")
        for status, count in stats['status_distribution'].items():
            logger.info(f"  - {status}: {count}")
    
    logger.info("\nFiles saved:")
    logger.info(f"  - Index: {FAISS_INDEX_PATH}")
    logger.info(f"  - Metadata: {METADATA_PATH}")
    logger.info(f"  - Chunk mapping: {CHUNK_MAPPING_PATH}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Embedding generation completed successfully!")
    logger.info("=" * 60)
    
    return vector_store


def test_search(vector_store: VectorStore, query: str, k: int = 5):
    """
    Test search functionality.
    
    Args:
        vector_store: Vector store instance
        query: Search query
        k: Number of results
    """
    logger.info(f"\nTesting search with query: '{query}'")
    logger.info("-" * 40)
    
    # Generate query embedding
    embedding_generator = EmbeddingGenerator(use_gpu=USE_GPU)
    query_embedding = embedding_generator.generate_query_embedding(query)
    
    # Search
    results = vector_store.search(query_embedding, k=k)
    
    # Display results
    if results:
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            title = metadata.get('BriefTitle', '') or metadata.get('OfficialTitle', '') or 'N/A'
            condition = metadata.get('Condition', '') or 'N/A'
            
            logger.info(f"\n{i}. NCT ID: {result['NCTId']}")
            logger.info(f"   Score: {result['score']:.4f}")
            logger.info(f"   Title: {title[:100]}...")
            logger.info(f"   Status: {metadata.get('OverallStatus', 'N/A')}")
            logger.info(f"   Condition: {condition[:100]}...")
    else:
        logger.info("No results found")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for clinical trials"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited samples"
    )
    
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10,
        help="Number of samples for test mode"
    )
    
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable document chunking"
    )
    
    parser.add_argument(
        "--search",
        type=str,
        help="Test search with a query after generation"
    )
    
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing embeddings instead of generating"
    )
    
    args = parser.parse_args()
    
    try:
        if args.load_existing:
            # Load existing embeddings
            logger.info("Loading existing embeddings...")
            vector_store = VectorStore()
            vector_store.load()
            logger.info("Embeddings loaded successfully")
        else:
            # Generate new embeddings
            vector_store = generate_embeddings_for_trials(
                use_chunking=not args.no_chunk,
                test_mode=args.test,
                test_samples=args.test_samples
            )
        
        # Test search if requested
        if args.search:
            test_search(vector_store, args.search)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()