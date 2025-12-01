# HHEM-2.1-Open Implementation Guide

Comprehensive guide for implementing Vectara's HHEM-2.1-Open hallucination evaluation model in RAG systems with LangChain/LangGraph.

## Overview

**HHEM-2.1-Open** (Hallucination Evaluation Model) is Vectara's open-source model for detecting hallucinations in LLM outputs. It takes premise-hypothesis pairs and outputs a consistency score between 0 and 1, where scores below 0.5 indicate hallucinations.

### Why HHEM over Alternatives

| Comparison | HHEM Advantage |
|------------|----------------|
| vs. LLM-as-judge | 50-60x faster (1.5s vs 35+s for GPT-4), more consistent, no echo chamber effects |
| vs. HHEM-1.0 | Supports up to 4096 tokens (vs 512), enabling more accurate RAG evaluation |
| vs. GPT-4 | Better F1 scores on RAGTruth and AggreFact benchmarks, significantly cheaper |

---

## Implementation Approaches

### Approach 1: AutoModel (Recommended for Production)

The simplest and fastest method using `transformers` directly:

```python
from transformers import AutoModelForSequenceClassification
import torch

# Step 1: Load model (one-time)
model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model',
    trust_remote_code=True  # CRITICAL: Required
)

# Optional: Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Step 2: Create premise-hypothesis pairs
pairs = [
    ("Paris is in France", "Paris is in France"),  # Consistent (~0.95)
    ("Paris is in France", "Paris is in Germany"),  # Hallucinated (~0.05)
    ("The Eiffel Tower was built in 1889", "It was constructed in 1889"),  # Consistent
]

# Step 3: Get scores
scores = model.predict(pairs)
# Returns: tensor([0.9500, 0.0500, 0.8700])

# Step 4: Interpret (threshold = 0.5)
for (premise, hypothesis), score in zip(pairs, scores):
    is_hallucinated = score < 0.5
    print(f"Score: {score:.4f} | Hallucinated: {is_hallucinated}")
```

**Key points:**
- Use `model.predict(pairs)` NOT `model(pairs)`
- Input format: `List[Tuple[str, str]]` - each tuple is `(premise, hypothesis)`
- Output: Tensor of floats between 0 and 1
- `trust_remote_code=True` is **REQUIRED**

### Approach 2: Pipeline (More Flexible)

Use for custom preprocessing or streaming:

```python
from transformers import pipeline, AutoTokenizer

# Create pipeline
classifier = pipeline(
    "text-classification",
    model='vectara/hallucination_evaluation_model',
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
    trust_remote_code=True,
    device=-1  # -1 for CPU
)

# Format inputs with the prompt template
prompt_template = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

pairs = [
    ("Paris is in France", "Paris is in France"),
    ("Paris is in France", "Paris is in Germany"),
]

formatted_pairs = [
    prompt_template.format(text1=p, text2=h)
    for p, h in pairs
]

results = classifier(formatted_pairs, top_k=None)

for formatted, result in zip(formatted_pairs, results):
    consistency_score = [
        item['score'] for item in result
        if item['label'] == 'consistent'
    ][0]
    is_hallucinated = consistency_score < 0.5
    print(f"Score: {consistency_score:.4f} | Hallucinated: {is_hallucinated}")
```

---

## Score Interpretation

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 0.0 - 0.3 | **Likely hallucination** | Reject or flag |
| 0.3 - 0.5 | **Uncertain/borderline** | Review or rewrite |
| 0.5 - 0.7 | **Likely consistent** | Generally safe |
| 0.7 - 1.0 | **Highly consistent** | Confident |

---

## Production-Ready Wrapper Class

```python
class HHEMEvaluator:
    def __init__(self, device='cpu'):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'vectara/hallucination_evaluation_model',
            trust_remote_code=True
        ).to(device).eval()

    def evaluate(self, premise: str, hypothesis: str, threshold: float = 0.5) -> dict:
        """Single pair evaluation"""
        with torch.no_grad():
            score = self.model.predict([(premise, hypothesis)]).item()

        return {
            'score': score,
            'is_hallucinated': score < threshold,
            'confidence': abs(score - threshold)
        }

    def batch_evaluate(self, pairs: list, threshold: float = 0.5) -> list:
        """Efficient batch evaluation for multiple pairs"""
        with torch.no_grad():
            scores = self.model.predict(pairs).tolist()

        return [
            {
                'premise': p,
                'hypothesis': h,
                'score': s,
                'is_hallucinated': s < threshold
            }
            for (p, h), s in zip(pairs, scores)
        ]

# Usage
evaluator = HHEMEvaluator(device='cpu')

# Single evaluation
result = evaluator.evaluate(
    premise="Paris is the capital of France",
    hypothesis="Paris is the capital of France"
)
print(f"Score: {result['score']:.4f}")

# Batch evaluation
results = evaluator.batch_evaluate([
    ("Context 1", "Answer 1"),
    ("Context 2", "Answer 2"),
])
```

---

## LangChain/LangGraph Integration

### Custom Evaluator Component

Wrap HHEM-2.1-Open as a LangChain `Runnable`:

```python
from langchain_core.runnables import Runnable

class HHEMRunnable(Runnable):
    def __init__(self):
        self.evaluator = HHEMEvaluator()

    def invoke(self, input: dict) -> dict:
        return self.evaluator.evaluate(
            premise=input['context'],
            hypothesis=input['answer']
        )
```

### Self-Correcting LangGraph Workflow

The most powerful pattern - a self-correcting RAG workflow:

```
Generate Node -> Evaluate Node -> (if hallucinated) -> Refine Node -> Finalize Node
                     |
                     v (if consistent)
                Finalize Node
```

**Nodes:**
- **Generate Node**: Creates initial LLM answer from retrieved context
- **Evaluate Node**: Uses HHEM to check if answer is hallucinated
- **Refine Node**: If hallucinated (score < 0.5), prompts LLM to regenerate more carefully
- **Finalize Node**: Returns final answer with quality metadata

This follows the Self-RAG pattern where the system evaluates its own outputs and adapts behavior accordingly.

---

## Architecture Patterns for RAG

For multi-agent setups with parallel retrieval:

1. Have each agent generate answers independently
2. Evaluate all answers with HHEM in batch mode
3. Filter out hallucinated answers (< 0.5 score)
4. Use consensus from high-confidence, non-hallucinated answers
5. Rerank based on both relevance and hallucination scores

### Key Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Batch Processing** | Process 16-32 pairs per batch for optimal throughput |
| **CPU-Only Mode** | HHEM runs efficiently on CPU (~1.5s per 2k tokens) |
| **Caching** | Implement LRU caching for repeated premise-hypothesis pairs |
| **Integration Point** | Place evaluation after generation, before returning to user |

---

## Performance & Memory

### Benchmarks (16GB RAM CPU System)

| Metric | Value |
|--------|-------|
| Single pair | ~50-100ms |
| Batch of 32 pairs | ~1.5s |
| Memory usage | <600MB |
| Throughput | ~20-30 pairs/second |

### Memory Requirements

- Model: ~600MB RAM at 32-bit precision
- Full process with overhead: ~1-2GB total
- Safe to run alongside small local retriever
- Processes 2k-token input in ~1.5s on modern x86 CPU

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| `model(pairs)` instead of `model.predict(pairs)` | Unexpected output | Always use `.predict()` |
| Missing `trust_remote_code=True` | Model fails to load | Add to `from_pretrained()` |
| Pairs in wrong format | Tokenization error | Use `List[Tuple[str, str]]` |
| No `.eval()` mode | Non-deterministic results | Add `model.eval()` after loading |
| Swapped premise/hypothesis | Reversed scores | Premise first, hypothesis second |

---

## Implementation Priority

1. **First**: Start with basic `evaluate_hallucination()` using AutoModel
2. **Second**: Wrap as `HHEMEvaluator` class, integrate into generation chain
3. **Third**: Build self-correcting LangGraph workflow for production
4. **Fourth**: Optimize with batch processing and caching

---

## References

- [HHEM-2.1-Open on HuggingFace](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Vectara HHEM-2.1 Blog Post](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model)
- [Vectara Hallucination Evaluation Docs](https://docs.vectara.com/docs/learn/hallucination-evaluation)
- [HuggingFace Transformers Pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
- [Self-RAG Pattern](https://www.projectpro.io/article/self-rag/1176)
