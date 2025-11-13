"""
Document loader module for loading documents with profiling and metadata enrichment.

This module handles document loading, profiling, and preparation for retrieval.
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from src.preprocessing.document_profiling import DocumentProfiler


class DocumentLoader:
    """
    Loads and profiles documents, enriching them with metadata for intelligent retrieval.

    Workflow:
    1. Load documents from various sources
    2. Profile each document using DocumentProfiler
    3. Enrich with metadata tags
    4. Prepare for vector store ingestion
    5. Calculate corpus-level statistics
    """

    def __init__(self):
        """Initialize the document loader with a profiler."""
        self.profiler = DocumentProfiler()
        self.documents_with_metadata: List[Document] = []
        self.corpus_stats: Dict = {}
        self.document_profiles: Dict[str, Dict] = {}

    def load_documents(
        self,
        documents: List[Document],
        verbose: bool = True
    ) -> Tuple[List[Document], Dict, Dict[str, Dict]]:
        """
        Load and profile documents, enriching them with metadata.

        Args:
            documents: List of LangChain Document objects
            verbose: Whether to print profiling progress

        Returns:
            Tuple of (enriched_documents, corpus_stats, document_profiles)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"DOCUMENT PROFILING")
            print(f"{'='*60}")
            print(f"Profiling {len(documents)} documents...\n")

        enriched_docs = []
        all_profiles = []

        for i, doc in enumerate(documents):
            # Get existing doc_id or generate one
            doc_id = doc.metadata.get("id", f"doc_{i}")

            # Profile the document
            profile, metadata = self.profiler.profile_document(
                doc.page_content,
                doc_id=doc_id
            )

            # Store profile
            self.document_profiles[doc_id] = profile
            all_profiles.append(profile)

            # Merge existing metadata with new profiling metadata
            enriched_metadata = {**doc.metadata, **metadata, "profile": profile}

            # Create enriched document
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=enriched_metadata
            )
            enriched_docs.append(enriched_doc)

            # Print profile summary
            if verbose:
                self._print_document_summary(i + 1, doc_id, profile, metadata)

        # Calculate corpus statistics
        doc_texts = [doc.page_content for doc in documents]
        self.corpus_stats = self.profiler.profile_corpus(doc_texts)

        if verbose:
            self._print_corpus_summary(self.corpus_stats)

        self.documents_with_metadata = enriched_docs

        return enriched_docs, self.corpus_stats, self.document_profiles

    def load_from_texts(
        self,
        texts: List[str],
        metadatas: List[Dict] = None,
        verbose: bool = True
    ) -> Tuple[List[Document], Dict, Dict[str, Dict]]:
        """
        Load documents from raw text strings.

        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts for each text
            verbose: Whether to print profiling progress

        Returns:
            Tuple of (enriched_documents, corpus_stats, document_profiles)
        """
        # Convert to Document objects
        if metadatas is None:
            metadatas = [{} for _ in texts]

        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        return self.load_documents(documents, verbose=verbose)

    def get_corpus_stats(self) -> Dict:
        """Get corpus-level statistics."""
        return self.corpus_stats

    def get_document_profile(self, doc_id: str) -> Dict:
        """Get profile for a specific document."""
        return self.document_profiles.get(doc_id, {})

    def get_all_profiles(self) -> Dict[str, Dict]:
        """Get all document profiles."""
        return self.document_profiles

    def _print_document_summary(self, doc_num: int, doc_id: str, profile: Dict, metadata: Dict):
        """Print summary of a single document's profile."""
        print(f"Document {doc_num} ({doc_id}):")
        print(f"  Type: {profile['type']}")
        print(f"  Length: {profile['length']} words")
        print(f"  Technical Density: {profile['technical_density']:.2f}")
        print(f"  Reading Level: {profile['reading_level']}")
        print(f"  Domain: {', '.join(profile['domain_tags'])}")
        print(f"  Best Strategy: {metadata['best_retrieval_strategy']}")
        print()

    def _print_corpus_summary(self, stats: Dict):
        """Print corpus-level statistics."""
        print(f"{'='*60}")
        print(f"CORPUS STATISTICS")
        print(f"{'='*60}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Average Length: {stats['avg_length']} words")
        print(f"Average Technical Density: {stats['avg_technical_density']:.2f}")
        print(f"\nDocument Types:")
        for doc_type, count in stats['document_types'].items():
            pct = (count / stats['total_documents']) * 100
            print(f"  - {doc_type}: {count} ({pct:.1f}%)")
        print(f"\nDomain Distribution:")
        for domain, count in stats['domain_distribution'].items():
            print(f"  - {domain}: {count}")
        print(f"\nPercentage with Code: {stats['pct_with_code']:.1f}%")
        print(f"Percentage with Math: {stats['pct_with_math']:.1f}%")
        print(f"{'='*60}\n")

    def filter_by_content_type(self, content_type: str) -> List[Document]:
        """
        Filter documents by content type.

        Args:
            content_type: The content type to filter by (e.g., 'tutorial', 'api_reference')

        Returns:
            List of documents matching the content type
        """
        return [
            doc for doc in self.documents_with_metadata
            if doc.metadata.get('content_type') == content_type
        ]

    def filter_by_domain(self, domain: str) -> List[Document]:
        """
        Filter documents by domain.

        Args:
            domain: The domain to filter by (e.g., 'machine_learning', 'rag')

        Returns:
            List of documents matching the domain
        """
        return [
            doc for doc in self.documents_with_metadata
            if doc.metadata.get('domain') == domain
        ]

    def filter_by_technical_level(self, level: str) -> List[Document]:
        """
        Filter documents by technical level.

        Args:
            level: The technical level (e.g., 'beginner', 'intermediate', 'advanced')

        Returns:
            List of documents matching the technical level
        """
        return [
            doc for doc in self.documents_with_metadata
            if doc.metadata.get('technical_level') == level
        ]

    def get_documents_for_strategy(self, strategy: str) -> List[Document]:
        """
        Get documents best suited for a specific retrieval strategy.

        Args:
            strategy: The retrieval strategy ('semantic', 'keyword', or 'hybrid')

        Returns:
            List of documents recommended for this strategy
        """
        return [
            doc for doc in self.documents_with_metadata
            if doc.metadata.get('best_retrieval_strategy') == strategy
        ]

    def get_summary(self) -> str:
        """Get a text summary of the loaded corpus."""
        if not self.corpus_stats:
            return "No documents loaded yet."

        stats = self.corpus_stats
        summary = f"""
Document Corpus Summary
=======================
Total Documents: {stats['total_documents']}
Average Length: {stats['avg_length']} words
Technical Density: {stats['avg_technical_density']:.2f}/1.0

Document Types: {', '.join(f"{k}({v})" for k, v in stats['document_types'].items())}
Top Domains: {', '.join(f"{k}({v})" for k, v in list(stats['domain_distribution'].items())[:3])}

{stats['pct_with_code']:.0f}% contain code
{stats['pct_with_math']:.0f}% contain math
"""
        return summary.strip()
