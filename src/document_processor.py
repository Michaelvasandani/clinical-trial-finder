"""Document processing and chunking for clinical trials."""

import logging
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..config import (
    FIELDS_TO_EMBED,
    METADATA_FIELDS,
    MAX_TOKENS_EMBED as MAX_TOKENS,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME as MODEL_NAME
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk clinical trial documents for embedding."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize document processor.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        
    def create_composite_document(self, trial: pd.Series) -> str:
        """
        Create a composite document from trial data.
        
        Args:
            trial: Series containing trial data
            
        Returns:
            Composite document string
        """
        sections = []
        
        # Process each section
        for section_name, fields in FIELDS_TO_EMBED.items():
            section_text = self._extract_section(trial, fields)
            if section_text:
                # Add section header for context
                if section_name == "title":
                    sections.append(f"Title: {section_text}")
                elif section_name == "summary":
                    sections.append(f"Summary: {section_text}")
                elif section_name == "eligibility":
                    sections.append(f"Eligibility: {section_text}")
                elif section_name == "medical":
                    sections.append(f"Medical Information: {section_text}")
                elif section_name == "outcomes":
                    sections.append(f"Outcomes: {section_text}")
        
        # Join sections with double newline
        document = "\n\n".join(sections)
        
        # Clean the document
        document = self._clean_text(document)
        
        return document
    
    def _extract_section(self, trial: pd.Series, fields: List[str]) -> str:
        """
        Extract and combine text from specified fields.
        
        Args:
            trial: Trial data
            fields: List of field names to extract
            
        Returns:
            Combined text from fields
        """
        texts = []
        
        for field in fields:
            if field in trial and pd.notna(trial[field]):
                text = str(trial[field]).strip()
                if text and text != "":
                    texts.append(text)
        
        return " ".join(texts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove very long repeated characters
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
        
        return text
    
    def chunk_document(
        self, 
        document: str, 
        max_tokens: int = MAX_TOKENS,
        overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """
        Chunk document into smaller pieces with overlap.
        
        Args:
            document: Document to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens
            
        Returns:
            List of document chunks
        """
        if not document:
            return []
        
        # Tokenize the document
        tokens = self.tokenizer.tokenize(document)
        
        # If document fits in one chunk, return as is
        if len(tokens) <= max_tokens:
            return [document]
        
        # Create chunks with overlap
        chunks = []
        sentences = self._split_into_sentences(document)
        
        current_chunk = []
        current_tokens = []
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            
            # Check if adding this sentence exceeds max tokens
            if current_tokens and len(current_tokens) + len(sentence_tokens) > max_tokens:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Take last few sentences for overlap
                    overlap_text = " ".join(current_chunk[-2:])  # Last 2 sentences
                    overlap_tokens = self.tokenizer.tokenize(overlap_text)
                    
                    if len(overlap_tokens) < overlap:
                        current_chunk = current_chunk[-2:]
                        current_tokens = overlap_tokens
                    else:
                        current_chunk = []
                        current_tokens = []
                else:
                    current_chunk = []
                    current_tokens = []
            
            current_chunk.append(sentence)
            current_tokens.extend(sentence_tokens)
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with nltk or spacy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Merge very short sentences
        merged_sentences = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) < 100:  # Merge if combined < 100 chars
                current = current + " " + sentence if current else sentence
            else:
                if current:
                    merged_sentences.append(current)
                current = sentence
        
        if current:
            merged_sentences.append(current)
        
        return merged_sentences
    
    def extract_metadata(self, trial: pd.Series) -> Dict:
        """
        Extract metadata fields from trial.
        
        Args:
            trial: Trial data
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        for field in METADATA_FIELDS:
            if field in trial:
                value = trial[field]
                # Convert to string if not null
                if pd.notna(value):
                    metadata[field] = str(value)
                else:
                    metadata[field] = ""
        
        return metadata
    
    def process_trials(
        self, 
        df: pd.DataFrame,
        chunk: bool = True
    ) -> Tuple[List[str], List[Dict], Dict]:
        """
        Process all trials into documents and metadata.
        
        Args:
            df: DataFrame with trial data
            chunk: Whether to chunk long documents
            
        Returns:
            Tuple of (documents, metadata_list, chunk_mapping)
        """
        all_documents = []
        all_metadata = []
        chunk_mapping = {}  # Maps chunk index to trial NCTId
        
        logger.info(f"Processing {len(df)} trials...")
        
        for idx, trial in df.iterrows():
            nct_id = trial["NCTId"]
            
            # Create composite document
            document = self.create_composite_document(trial)
            
            # Extract metadata
            metadata = self.extract_metadata(trial)
            
            if chunk:
                # Chunk the document
                chunks = self.chunk_document(document)
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    # Add chunk to documents
                    all_documents.append(chunk_text)
                    
                    # Create metadata for this chunk
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_idx
                    chunk_metadata["total_chunks"] = len(chunks)
                    all_metadata.append(chunk_metadata)
                    
                    # Map chunk to trial
                    chunk_mapping[len(all_documents) - 1] = nct_id
            else:
                # Add whole document
                all_documents.append(document)
                all_metadata.append(metadata)
                chunk_mapping[len(all_documents) - 1] = nct_id
        
        logger.info(f"Created {len(all_documents)} document chunks from {len(df)} trials")
        
        return all_documents, all_metadata, chunk_mapping