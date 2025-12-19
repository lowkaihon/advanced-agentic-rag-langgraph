# src/advanced_agentic_rag_langgraph/preprocessing/json_chunk_loader.py

from pathlib import Path
from typing import List, Tuple
import json
from langchain_core.documents import Document


class MarkerJSONLoader:
    """Load pre-chunked documents from Marker-processed JSON files."""

    def __init__(self, json_dir: str | Path = None):
        if json_dir is None:
            import advanced_agentic_rag_langgraph
            project_root = Path(advanced_agentic_rag_langgraph.__file__).parent.parent.parent
            json_dir = project_root / "evaluation" / "corpus_chunks" / "marker_json_v2"
        self.json_dir = Path(json_dir)

    def load_all(self, verbose: bool = True) -> Tuple[List[Document], List[Document]]:
        """Load all JSON files from directory.

        Returns:
            Tuple of (full_documents, chunks)
            - full_documents: One Document per source (for profiling)
            - chunks: All individual chunks with source metadata
        """
        full_docs = []
        all_chunks = []

        json_files = sorted(self.json_dir.glob("*.json"))

        if verbose:
            print(f"Found {len(json_files)} JSON files in {self.json_dir}")

        for json_file in json_files:
            source_full_doc, source_chunks = self._load_single_json(json_file)
            full_docs.append(source_full_doc)
            all_chunks.extend(source_chunks)

        if verbose:
            print(f"Loaded {len(full_docs)} documents, {len(all_chunks)} chunks")

        return full_docs, all_chunks

    def _load_single_json(self, json_path: Path) -> Tuple[Document, List[Document]]:
        """Load single JSON file.

        Returns:
            Tuple of (full_document, chunks)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        source = data.get('source', json_path.stem)

        # Filter to text chunks only (skip _table_ and _figure_)
        text_chunks = [
            c for c in data.get('chunks', [])
            if '_chunk_' in c.get('id', '')
        ]

        # Create chunk Documents
        chunk_docs = []
        for chunk in text_chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'id': chunk['id'],
                    'source': source,
                    'source_type': 'marker_json',
                    'chunk_index': chunk.get('metadata', {}).get('chunk_index', 0),
                    'char_count': chunk.get('metadata', {}).get('char_count', len(chunk['content'])),
                }
            )
            chunk_docs.append(doc)

        # Use markdown field for full document (already contains complete content)
        full_content = data.get('markdown', '')
        full_doc = Document(
            page_content=full_content,
            metadata={
                'id': source,
                'source': source,
                'source_type': 'marker_json',
                'total_chunks': len(chunk_docs),
            }
        )

        return full_doc, chunk_docs
