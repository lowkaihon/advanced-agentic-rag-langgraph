# Golden Dataset for RAG Evaluation

This directory contains the golden dataset for offline evaluation of the Advanced Agentic RAG system.

**Helper Tool:** Use `tests/utils/create_golden_dataset_helper.py` to validate chunk IDs or create new examples.

## Overview

The golden dataset consists of **20 manually-curated examples** designed to comprehensively test retrieval and generation quality across multiple dimensions.

### Dataset Composition

- **Total Examples**: 20
- **Core Papers**: 3 (Attention Is All You Need, BERT, DDPM) - 15 examples
- **Cross-Document Examples**: 5 examples spanning multiple papers

### Coverage

**By Difficulty:**
- Easy: 6 examples (30%) - Simple factual lookups
- Medium: 10 examples (50%) - Conceptual understanding
- Hard: 4 examples (20%) - Multi-hop reasoning, procedural

**By Query Type:**
- Factual: Questions seeking specific facts or numbers
- Conceptual: Questions requiring understanding of concepts
- Procedural: Questions about processes or workflows
- Comparative: Questions comparing different approaches

**By Domain:**
- NLP: Natural language processing (Transformers, BERT)
- Computer Vision: Vision models (ViT)
- Generative Models: Diffusion models, GANs
- RAG: Meta-evaluation (RAG reading about RAG)
- Cross-Domain: Multi-domain queries

## Files

- `golden_set.json` - The main golden dataset (20 examples)
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
from src.evaluation import GoldenDatasetManager

manager = GoldenDatasetManager("test_datasets/golden_set.json")
manager.print_statistics()
```

**Run evaluation programmatically:**
```python
from src.evaluation import GoldenDatasetManager, evaluate_on_golden_dataset
from src.orchestration.graph import advanced_rag_graph

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
from src.evaluation import GoldenDatasetManager

manager = GoldenDatasetManager("test_datasets/golden_set.json")

# Check all examples
for example in manager.dataset:
    is_valid, errors = manager.validate_example(example)
    if not is_valid:
        print(f"Example {example['id']}: {errors}")

# Verify chunk IDs exist in corpus
from src.core.config import setup_retriever
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
