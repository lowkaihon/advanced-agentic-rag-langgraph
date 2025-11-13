"""
PDF document loader with intelligent chunking for RAG.

This module provides utilities for loading PDF documents, chunking them
appropriately, and preparing them for the document profiling pipeline.
"""

import os
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFDocumentLoader:
    """
    Loads PDF documents and chunks them for RAG.

    Features:
    - Page-by-page PDF text extraction using PyPDFLoader
    - Intelligent text chunking with RecursiveCharacterTextSplitter
    - Preserves document structure (paragraphs, sentences)
    - Adds rich metadata for tracking
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize PDF loader with chunking configuration.

        Args:
            chunk_size: Maximum characters per chunk (default: 1000)
            chunk_overlap: Characters overlap between chunks (default: 200)
            separators: List of split separators (default: paragraph/sentence boundaries)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators prioritize document structure
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                " ",     # Word boundaries
                ""       # Character-level fallback
            ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def load_pdf(
        self,
        pdf_path: str,
        source_name: Optional[str] = None,
        verbose: bool = True
    ) -> List[Document]:
        """
        Load and chunk a PDF document.

        Args:
            pdf_path: Path to the PDF file
            source_name: Optional name for the source (defaults to filename)
            verbose: Whether to print loading progress

        Returns:
            List of Document objects (chunked and enriched with metadata)
        """
        # Validate file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Extract source name from path if not provided
        if source_name is None:
            source_name = os.path.basename(pdf_path)

        if verbose:
            print(f"\n{'='*60}")
            print(f"LOADING PDF: {source_name}")
            print(f"{'='*60}")
            print(f"Path: {pdf_path}")

        # Load PDF pages
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()

        if verbose:
            print(f"Extracted {len(pages)} pages from PDF")

        # Chunk the documents
        chunks = self.text_splitter.split_documents(pages)

        if verbose:
            print(f"Created {len(chunks)} chunks")
            print(f"Chunk size: {self.chunk_size} characters")
            print(f"Chunk overlap: {self.chunk_overlap} characters")

        # Enrich chunks with metadata
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Preserve existing metadata and add new fields
            chunk.metadata.update({
                "id": f"{source_name}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source_name,
                "source_type": "pdf",
            })
            enriched_chunks.append(chunk)

        if verbose:
            # Show sample chunk
            if enriched_chunks:
                sample = enriched_chunks[0]
                print(f"\nSample chunk (first 200 chars):")
                print(f"{sample.page_content[:200]}...")
                print(f"\nMetadata: {sample.metadata}")
            print(f"{'='*60}\n")

        return enriched_chunks

    def load_multiple_pdfs(
        self,
        pdf_paths: List[str],
        verbose: bool = True
    ) -> List[Document]:
        """
        Load and chunk multiple PDF documents.

        Args:
            pdf_paths: List of paths to PDF files
            verbose: Whether to print loading progress

        Returns:
            Combined list of Document objects from all PDFs
        """
        all_chunks = []

        for pdf_path in pdf_paths:
            chunks = self.load_pdf(pdf_path, verbose=verbose)
            all_chunks.extend(chunks)

        if verbose:
            print(f"\nTotal documents from {len(pdf_paths)} PDFs: {len(all_chunks)}")

        return all_chunks

    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunked documents.

        Args:
            chunks: List of chunked documents

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}

        chunk_lengths = [len(chunk.page_content) for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths),
            "chunk_size_config": self.chunk_size,
            "chunk_overlap_config": self.chunk_overlap,
        }

        return stats

    def print_statistics(self, chunks: List[Document]):
        """Print formatted statistics about chunks."""
        stats = self.get_chunk_statistics(chunks)

        if not stats:
            print("No chunks to analyze")
            return

        print(f"\n{'='*60}")
        print(f"CHUNK STATISTICS")
        print(f"{'='*60}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Average length: {stats['avg_chunk_length']:.0f} characters")
        print(f"Min length: {stats['min_chunk_length']} characters")
        print(f"Max length: {stats['max_chunk_length']} characters")
        print(f"Total characters: {stats['total_characters']:,}")
        print(f"\nConfiguration:")
        print(f"  Chunk size: {stats['chunk_size_config']}")
        print(f"  Chunk overlap: {stats['chunk_overlap_config']}")
        print(f"{'='*60}\n")


# Convenience function for quick loading
def load_pdf_for_rag(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True
) -> List[Document]:
    """
    Quick utility to load a PDF with default RAG settings.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters overlap between chunks
        verbose: Print loading progress

    Returns:
        List of chunked Document objects
    """
    loader = PDFDocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return loader.load_pdf(pdf_path, verbose=verbose)
