# Golden Dataset for RAG Evaluation

This directory contains the golden datasets for offline evaluation of the Advanced Agentic RAG system.

## Dataset Versions

### v2 (Current) - Marker PDF Processing
- **Files:** `golden_set_standard_v2.json`, `golden_set_hard_v2.json`
- **Preprocessing:** [Marker](https://github.com/VikParuchuri/marker) with Google Colab GPU (OCR-capable, not required for digital-born PDFs)
- **Chunks:** `corpus_chunks/marker_json_v2/` (1,295 text chunks from 10 documents)
- **Chunk ID format:** `{source}_chunk_{index}` (e.g., `Attention Is All You Need.pdf_chunk_20`)
- **Benefits:**
  - Table extraction (~96% vs 0% with PyMuPDF)
  - Figure extraction with Vision LLM descriptions (GPT-4o-mini)
  - Layout-aware processing with clean markdown output
- **Notebook:** `src/advanced_agentic_rag_langgraph/preprocessing/marker_preprocessing.ipynb`

> **Note on Figures:** Figure descriptions are extracted via GPT-4o-mini vision and stored in chunk metadata. They are excluded from the default retrieval index to maintain golden set evaluation consistency. The loader supports `include_figures=True` for multimodal retrieval use cases.

### v1 (Legacy) - PyMuPDF Processing
- **Files:** `golden_set_standard.json`, `golden_set_hard.json`
- **Preprocessing:** PyMuPDF with RecursiveCharacterTextSplitter
- **Note:** Chunk IDs differ from v2; use v2 for current evaluations

## Two-Tier Evaluation Strategy

The evaluation uses a **two-tier dataset approach** to demonstrate both baseline competence and advanced capabilities.

### Standard Dataset (`golden_set_standard.json`) - 20 questions
- **Purpose:** Production-ready baseline performance
- **Configuration:** k_final=4

### Hard Dataset (`golden_set_hard.json`) - 10 questions
- **Purpose:** Multi-hop reasoning across documents
- **Configuration:** k_final=6

## Dataset Schema

Each example contains:

```json
{
  "id": "transformer_time_complexity",
  "question": "What is the time complexity of self-attention in the Transformer?",
  "ground_truth_answer": "In the Transformer, self-attention layers have a per-layer computational complexity of O(n^2 * d)...",
  "relevant_doc_ids": [
    "Attention Is All You Need.pdf_chunk_20",
    "Attention Is All You Need.pdf_chunk_23"
  ],
  "relevance_grades": {
    "Attention Is All You Need.pdf_chunk_20": 3,
    "Attention Is All You Need.pdf_chunk_23": 2
  },
  "source_document": "Attention Is All You Need.pdf",
  "difficulty": "easy",
  "query_type": "factual",
  "domain": "nlp",
  "expected_strategy": "keyword",
  "expected_chunks": 1
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `question` | User query to test |
| `ground_truth_answer` | Expected correct answer (100-200 words) |
| `relevant_doc_ids` | Chunk IDs that should be retrieved |
| `relevance_grades` | Graded relevance (0-3 scale) for nDCG |
| `difficulty` | easy, medium, or hard |
| `query_type` | factual, conceptual, procedural, or comparative |
| `expected_strategy` | semantic, keyword, or hybrid |
| `expected_chunks` | Expected number of relevant chunks needed |

### Relevance Grades (0-3 Scale)

| Grade | Meaning |
|-------|---------|
| 3 | Highly relevant - contains direct answer |
| 2 | Relevant - supporting information |
| 1 | Marginally relevant - background info |
| 0 | Not relevant |

## Evaluation Metrics

**Retrieval:** F1@K, Precision@K, Recall@K, MRR, nDCG@K

**Generation:** Groundedness (HHEM-based), Semantic Similarity, Factual Accuracy, Completeness
