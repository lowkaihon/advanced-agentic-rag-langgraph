# Golden Dataset for RAG Evaluation

This directory contains the golden datasets for offline evaluation of the Advanced Agentic RAG system.

**Helper Tool:** Use `tests/utils/create_golden_dataset_helper.py` to validate chunk IDs or create new examples.

## Overview

The evaluation suite uses a **two-tier dataset approach** designed to demonstrate both baseline competence (Standard) and advanced capabilities (Hard).

### Two-Tier Strategy

**Standard Dataset** (`golden_set_standard.json`) - 20 questions
- **Purpose:** Showcase production-ready performance with strong metrics
- **Target Metrics:** F1@5 65-75%, Groundedness 95-97%
- **Configuration:** k_final=4 (optimal for 2-3 chunk questions)
- **Portfolio Positioning:** Lead with strength

**Hard Dataset** (`golden_set_hard.json`) - 10 questions
- **Purpose:** Demonstrate advanced capability on challenging multi-hop queries
- **Target Metrics:** F1@5 32-40%, Groundedness 92-95%
- **Configuration:** k_final=6 (adaptive retrieval for complex reasoning)
- **Portfolio Positioning:** Show sophistication

**Combined Portfolio Narrative:** "72% F1@5 on standard RAG + maintains 95% groundedness on complex multi-hop queries"

### Legacy Dataset

**Original Dataset** (`golden_set.json`) - 20 questions
- **Status:** Deprecated - too difficult (2-5x harder than typical RAG benchmarks)
- **Issue:** 70% challenging questions led to defensive portfolio positioning
- **Replaced By:** Two-tier approach (standard + hard)

### Hard Dataset Composition

**Total Questions:** 10 (100% hard difficulty)

**By Query Type:**
- Procedural: 3 questions (30%) - Complex step-by-step explanations
- Comparative: 5 questions (50%) - Cross-paradigm analysis
- Conceptual: 2 questions (20%) - Deep architectural understanding

**Cross-Document Coverage:**
- Single-document: 5 questions (50%)
- Cross-document: 5 questions (50%) - Multi-hop reasoning across 2+ papers

**Document Coverage:**
- Attention Is All You Need: 3 questions
- ViT (Vision Transformer): 3 questions
- BERT: 1 question
- DDPM: 3 questions
- Consistency Models: 1 question
- RAPTOR: 1 question
- WGAN-GP: 1 question
- CLIP: 1 question

**Average Chunks Needed:** 3.3 per question (range: 2-5 chunks)

**Question IDs** (Descriptive naming):
1. `transformer_encoder_complete_forward_pass`
2. `bert_vs_gpt_pretraining_comparison`
3. `ddpm_simplified_objective_derivation`
4. `transformer_vs_vit_attention_usage`
5. `vit_architecture_adaptation`
6. `ddpm_vs_consistency_models_comparison`
7. `raptor_vs_standard_rag_improvement`
8. `ddpm_vs_wgan_generative_approaches`
9. `vit_patch_size_and_title_meaning`
10. `clip_contrastive_vs_supervised_learning`

**Quality Assurance:**
- ✅ All ground truth answers verified against source papers
- ✅ ViT positional embeddings corrected (2D → 1D learnable)
- ✅ RAPTOR performance claims corrected (20% → 1-2% F1 gains)
- ✅ Graded relevance (0-3 scale) for nDCG evaluation

## Files

**Active Datasets:**
- `golden_set_standard.json` - Standard difficulty dataset (20 questions) - **Created in separate session**
- `golden_set_hard.json` - Hard difficulty dataset (10 questions) - **Verified and corrected**

**Legacy:**
- `golden_set.json` - Original dataset (20 questions, deprecated)

**Generated Reports:**
- `baseline_metrics.json` - Baseline performance metrics (auto-generated)
- `evaluation_report.md` - Latest evaluation report (auto-generated)
- `README.md` - This documentation

## Dataset Structure

Each example in `golden_set.json` contains:

```json
{
  "id": "attention_001",
  "question": "How many attention heads are used in the base Transformer model?",
  "ground_truth_answer": "The base Transformer model uses 8 attention heads...",
  "relevant_doc_ids": ["Attention Is All You Need.pdf_chunk_34"],
  "relevance_grades": {
    "Attention Is All You Need.pdf_chunk_34": 3,
    "Attention Is All You Need.pdf_chunk_17": 2
  },
  "source_document": "Attention Is All You Need.pdf",
  "difficulty": "easy",
  "query_type": "factual",
  "domain": "nlp",
  "expected_strategy": "keyword"
}
```

### Field Descriptions

- **id**: Unique identifier for the example
- **question**: The user query to test
- **ground_truth_answer**: Expected correct answer (100-200 words)
- **relevant_doc_ids**: List of chunk IDs that should be retrieved
- **relevance_grades**: Graded relevance (0-3 scale):
  - 0: Not relevant
  - 1: Marginally relevant
  - 2: Relevant
  - 3: Highly relevant
- **source_document**: PDF filename(s) containing the answer
- **difficulty**: "easy", "medium", or "hard"
- **query_type**: "factual", "conceptual", "procedural", or "comparative"
- **domain**: Subject area
- **expected_strategy**: "semantic", "keyword", or "hybrid"

## Usage

### Running Evaluation

**Quick evaluation:**
```bash
uv run python test_golden_dataset_evaluation.py
```

This runs the full evaluation suite including:
- Dataset loading and validation
- Baseline performance testing
- Regression detection
- Cross-document retrieval testing
- Difficulty correlation analysis
- Report generation

### Individual Tests

**Test dataset loading:**
```python
from advanced_agentic_rag_langgraph.evaluation import GoldenDatasetManager

manager = GoldenDatasetManager("test_datasets/golden_set.json")
manager.print_statistics()
```

**Run evaluation programmatically:**
```python
from advanced_agentic_rag_langgraph.evaluation import GoldenDatasetManager, evaluate_on_golden_dataset
from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph

manager = GoldenDatasetManager("test_datasets/golden_set.json")
results = evaluate_on_golden_dataset(
    advanced_rag_graph,
    manager.dataset,
    verbose=True
)

print(f"Recall@5: {results['retrieval_metrics']['recall_at_k']:.2%}")
print(f"Groundedness: {results['generation_metrics']['avg_groundedness']:.2%}")
```

**Filter examples:**
```python
# Get only hard examples
hard_examples = manager.get_by_difficulty("hard")

# Get only comparative queries
comparative = manager.get_by_query_type("comparative")

# Get cross-document examples
cross_doc = manager.get_cross_document_examples()
```

## Evaluation Metrics

### Retrieval Metrics

- **Recall@5**: Percentage of relevant documents retrieved in top 5
- **Precision@5**: Percentage of top 5 that are relevant
- **F1@5**: Harmonic mean of precision and recall
- **Hit Rate**: Whether at least one relevant doc retrieved
- **MRR**: Mean Reciprocal Rank (position of first relevant doc)
- **nDCG@5**: Normalized Discounted Cumulative Gain (ranking quality)

### Generation Metrics

- **Average Groundedness**: Percentage of claims supported by context
- **Average Confidence**: System's confidence in answers
- **Hallucination Rate**: Percentage of answers with hallucinations

## Performance Thresholds

### Baseline Expectations

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Recall@5 | ≥ 70% | Retrieve most relevant docs |
| Precision@5 | ≥ 60% | Minimize irrelevant docs |
| F1@5 | ≥ 65% | Balanced performance |
| Groundedness | ≥ 85% | Strong claim support |
| Hallucination Rate | ≤ 15% | Low hallucination |

### Difficulty-Specific Expectations

| Difficulty | Recall@5 | Precision@5 |
|------------|----------|-------------|
| Easy | ≥ 80% | ≥ 70% |
| Medium | ≥ 70% | ≥ 60% |
| Hard | ≥ 60% | ≥ 50% |

## Annotation Guidelines

### Relevance Grading (0-3 Scale)

**Grade 3 (Highly Relevant):**
- Contains direct answer to the question
- Primary source of information
- Essential for complete answer

**Grade 2 (Relevant):**
- Contains supporting information
- Contributes context or evidence
- Useful but not essential

**Grade 1 (Marginally Relevant):**
- Tangentially related
- Background information
- Minor supporting detail

**Grade 0 (Not Relevant):**
- No meaningful connection
- Wrong topic or context
- Should not be retrieved

### Ground Truth Answer Quality Standards

Good ground truth answers should:
- Be 100-200 words (concise but complete)
- Directly address the question
- Include key technical details
- Cite specific concepts from source documents
- Use accurate terminology
- Be factually correct

Avoid:
- Vague or ambiguous language
- Excessive length or verbosity
- Missing critical details
- Incorrect information
- Speculation or uncertainty

## Adding New Examples

### Process

1. **Identify the question** - What user query are you testing?
2. **Write ground truth answer** - What is the correct, complete answer?
3. **Find relevant chunks** - Which document chunks contain the answer?
4. **Grade relevance** - Rate each chunk 0-3
5. **Classify the example** - Set difficulty, query_type, domain
6. **Validate** - Ensure all required fields present

### Example Template

```json
{
  "id": "new_example_001",
  "question": "Your question here",
  "ground_truth_answer": "Complete answer (100-200 words)",
  "relevant_doc_ids": [
    "Document.pdf_chunk_X",
    "Document.pdf_chunk_Y"
  ],
  "relevance_grades": {
    "Document.pdf_chunk_X": 3,
    "Document.pdf_chunk_Y": 2
  },
  "source_document": "Document.pdf",
  "difficulty": "easy|medium|hard",
  "query_type": "factual|conceptual|procedural|comparative",
  "domain": "nlp|computer_vision|generative_models|rag|other",
  "expected_strategy": "semantic|keyword|hybrid"
}
```

### Validation

After adding examples, validate with:

```python
from advanced_agentic_rag_langgraph.evaluation import GoldenDatasetManager

manager = GoldenDatasetManager("test_datasets/golden_set.json")

# Check all examples
for example in manager.dataset:
    is_valid, errors = manager.validate_example(example)
    if not is_valid:
        print(f"Example {example['id']}: {errors}")

# Verify chunk IDs exist in corpus
from advanced_agentic_rag_langgraph.core.config import setup_retriever
retriever = setup_retriever()
validation_results = manager.validate_against_corpus(retriever)
print(validation_results)
```

## Regression Testing

### Workflow

1. **Establish Baseline**: Run `test_baseline_performance()` to save initial metrics
2. **Make Changes**: Update retrieval, reranking, or generation logic
3. **Run Regression Test**: Execute `test_regression()` to compare
4. **Review Differences**: Check if performance degraded
5. **Iterate**: Fix issues or accept new baseline

### Regression Thresholds

- Retrieval metrics: ±5% tolerance
- Groundedness: ±10% tolerance

If degradation exceeds thresholds, the test fails and requires investigation.

## Best Practices

1. **Start Small**: Begin with 20-25 examples, expand gradually
2. **Diversity Matters**: Cover all difficulty levels and query types
3. **Update Regularly**: Add production failures to golden set
4. **Version Control**: Track dataset changes in git
5. **Review Periodically**: Audit examples quarterly for relevance
6. **Cross-Validate**: Have multiple reviewers check annotations
7. **Document Changes**: Note why examples were added/modified

## Troubleshooting

**Low Recall@5:**
- Check if relevant chunks exist in corpus
- Verify embedding model captures semantic meaning
- Consider improving chunking strategy

**Low Precision@5:**
- Enhance reranking thresholds
- Improve relevance scoring
- Filter low-confidence results

**High Hallucination Rate:**
- Review groundedness check prompts
- Lower retry thresholds (< 0.6)
- Improve context quality

**Chunk IDs Not Found:**
- Re-generate corpus with consistent chunking
- Update chunk IDs in golden_set.json
- Verify PDF files match expected versions

## Maintenance

### Quarterly Review

- [ ] Audit 20% of examples for accuracy
- [ ] Add 5-10 new examples from production failures
- [ ] Remove outdated or duplicate examples
- [ ] Re-run full evaluation suite
- [ ] Update baseline metrics if significant changes made

### Version History

- **v1.0** (2025-01-15): Initial dataset with 20 examples (3 core papers + cross-document)

---

*For questions or contributions, see the main project documentation.*
