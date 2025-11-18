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

    Features: Page extraction, intelligent chunking, structure preservation, metadata tracking.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = [
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
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
        """Load and chunk a PDF document."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if source_name is None:
            source_name = os.path.basename(pdf_path)

        if verbose:
            print(f"\n{'='*60}")
            print(f"LOADING PDF: {source_name}")
            print(f"{'='*60}")
            print(f"Path: {pdf_path}")

        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()

        if verbose:
            print(f"Extracted {len(pages)} pages from PDF")

        chunks = self.text_splitter.split_documents(pages)

        if verbose:
            print(f"Created {len(chunks)} chunks")
            print(f"Chunk size: {self.chunk_size} characters")
            print(f"Chunk overlap: {self.chunk_overlap} characters")

        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "id": f"{source_name}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source_name,
                "source_type": "pdf",
            })
            enriched_chunks.append(chunk)

        if verbose:
            if enriched_chunks:
                sample = enriched_chunks[0]
                print(f"\nSample chunk (first 200 chars):")
                print(f"{sample.page_content[:200]}...")
                print(f"\nMetadata: {sample.metadata}")
            print(f"{'='*60}\n")

        return enriched_chunks

    def load_pdf_full_document(
        self,
        pdf_path: str,
        source_name: Optional[str] = None,
        verbose: bool = True
    ) -> Document:
        """Load a PDF as a single full document (WITHOUT chunking)."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if source_name is None:
            source_name = os.path.basename(pdf_path)

        if verbose:
            print(f"Loading full document: {source_name}")

        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()

        full_text = "\n\n".join([page.page_content for page in pages])

        full_document = Document(
            page_content=full_text,
            metadata={
                "source": source_name,
                "source_type": "pdf",
                "page_count": len(pages),
                "char_count": len(full_text),
            }
        )

        if verbose:
            print(f"  Pages: {len(pages)}, Characters: {len(full_text):,}")

        return full_document

    def load_multiple_pdfs(
        self,
        pdf_paths: List[str],
        verbose: bool = True
    ) -> List[Document]:
        """Load and chunk multiple PDF documents."""
        all_chunks = []

        for pdf_path in pdf_paths:
            chunks = self.load_pdf(pdf_path, verbose=verbose)
            all_chunks.extend(chunks)

        if verbose:
            print(f"\nTotal documents from {len(pdf_paths)} PDFs: {len(all_chunks)}")

        return all_chunks

    def load_multiple_pdfs_full_documents(
        self,
        pdf_paths: List[str],
        verbose: bool = True
    ) -> List[Document]:
        """Load multiple PDFs as full documents (WITHOUT chunking)."""
        full_documents = []

        if verbose:
            print(f"\nLoading {len(pdf_paths)} PDFs as full documents (no chunking)...")

        for pdf_path in pdf_paths:
            doc = self.load_pdf_full_document(pdf_path, verbose=verbose)
            full_documents.append(doc)

        if verbose:
            total_chars = sum(len(doc.page_content) for doc in full_documents)
            print(f"\nLoaded {len(full_documents)} full documents, {total_chars:,} total characters\n")

        return full_documents

    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """Get statistics about the chunked documents."""
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
    """Quick utility to load a PDF with default RAG settings."""
    loader = PDFDocumentLoader(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return loader.load_pdf(pdf_path, verbose=verbose)
