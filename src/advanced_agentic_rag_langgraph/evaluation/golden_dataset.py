"""
Golden Dataset Management and Evaluation for RAG Systems.

This module provides utilities for managing golden datasets and conducting
offline evaluation of RAG pipelines using ground truth data.
"""

import json
import os
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .retrieval_metrics import calculate_answer_relevance


class GoldenDatasetManager:
    """
    Manage and validate golden datasets for RAG evaluation.

    Features: Load/save datasets, filter by difficulty/query type/domain,
    validate structure, verify chunk IDs, generate statistics.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = []

        if os.path.exists(dataset_path):
            self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Dict]:
        """Load and validate golden dataset from JSON file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Validate each example
        validation_errors = []
        for i, example in enumerate(dataset):
            is_valid, errors = self.validate_example(example)
            if not is_valid:
                validation_errors.append(f"Example {i} ({example.get('id', 'unknown')}): {', '.join(errors)}")

        if validation_errors:
            print(f"Warning: Found {len(validation_errors)} validation errors:")
            for error in validation_errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(validation_errors) > 5:
                print(f"  ... and {len(validation_errors) - 5} more")

        print(f"Loaded {len(dataset)} examples from golden dataset")
        return dataset

    def save_dataset(self, dataset: List[Dict]):
        """Save dataset to JSON file with validation."""
        for example in dataset:
            is_valid, errors = self.validate_example(example)
            if not is_valid:
                raise ValueError(f"Invalid example {example.get('id')}: {', '.join(errors)}")

        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(dataset)} examples to {self.dataset_path}")

    def validate_example(self, example: Dict) -> Tuple[bool, List[str]]:
        """
        Validate single example structure and completeness.

        Returns: (is_valid, list_of_errors)
        """
        errors = []

        required_fields = [
            "id", "question", "ground_truth_answer", "relevant_doc_ids",
            "source_document", "difficulty", "query_type", "domain", "expected_strategy"
        ]

        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")

        valid_difficulties = ["easy", "medium", "hard"]
        if example.get("difficulty") not in valid_difficulties:
            errors.append(f"Invalid difficulty: {example.get('difficulty')}")

        valid_query_types = ["factual", "conceptual", "procedural", "comparative"]
        if example.get("query_type") not in valid_query_types:
            errors.append(f"Invalid query_type: {example.get('query_type')}")

        valid_strategies = ["semantic", "keyword", "hybrid"]
        if example.get("expected_strategy") not in valid_strategies:
            errors.append(f"Invalid expected_strategy: {example.get('expected_strategy')}")

        answer = example.get("ground_truth_answer", "")
        if len(answer) < 50:
            errors.append("Ground truth answer too short (< 50 chars)")
        if len(answer) > 2000:
            errors.append("Ground truth answer too long (> 2000 chars)")

        if not isinstance(example.get("relevant_doc_ids"), list):
            errors.append("relevant_doc_ids must be a list")

        if "relevance_grades" in example:
            relevance_grades = example["relevance_grades"]
            doc_ids = set(example.get("relevant_doc_ids", []))

            graded_ids = set(relevance_grades.keys())
            if not graded_ids.issubset(doc_ids):
                errors.append("relevance_grades contains doc_ids not in relevant_doc_ids")

            for doc_id, grade in relevance_grades.items():
                if not isinstance(grade, int) or grade not in [0, 1, 2, 3]:
                    errors.append(f"Invalid relevance grade for {doc_id}: {grade} (must be 0-3)")

        return (len(errors) == 0, errors)

    def add_example(self, example: Dict) -> bool:
        """Add new example to dataset with validation."""
        is_valid, errors = self.validate_example(example)
        if not is_valid:
            print(f"Cannot add invalid example: {', '.join(errors)}")
            return False

        self.dataset.append(example)
        return True

    def get_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Filter examples by difficulty level."""
        return [ex for ex in self.dataset if ex.get("difficulty") == difficulty]

    def get_by_query_type(self, query_type: str) -> List[Dict]:
        """Filter examples by query type."""
        return [ex for ex in self.dataset if ex.get("query_type") == query_type]

    def get_by_domain(self, domain: str) -> List[Dict]:
        """Filter examples by domain."""
        return [ex for ex in self.dataset if ex.get("domain") == domain]

    def get_cross_document_examples(self) -> List[Dict]:
        """Get examples that involve multiple source documents."""
        return [
            ex for ex in self.dataset
            if isinstance(ex.get("source_document"), list) and len(ex["source_document"]) > 1
        ]

    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        if not self.dataset:
            return {}

        difficulty_counts = defaultdict(int)
        for ex in self.dataset:
            difficulty_counts[ex.get("difficulty", "unknown")] += 1

        query_type_counts = defaultdict(int)
        for ex in self.dataset:
            query_type_counts[ex.get("query_type", "unknown")] += 1

        domain_counts = defaultdict(int)
        for ex in self.dataset:
            domain_counts[ex.get("domain", "unknown")] += 1

        strategy_counts = defaultdict(int)
        for ex in self.dataset:
            strategy_counts[ex.get("expected_strategy", "unknown")] += 1

        cross_doc_count = len(self.get_cross_document_examples())

        source_doc_counts = defaultdict(int)
        for ex in self.dataset:
            source_doc = ex.get("source_document", "unknown")
            if isinstance(source_doc, list):
                for doc in source_doc:
                    source_doc_counts[doc] += 1
            else:
                source_doc_counts[source_doc] += 1

        stats = {
            "total_examples": len(self.dataset),
            "difficulty_distribution": dict(difficulty_counts),
            "query_type_distribution": dict(query_type_counts),
            "domain_distribution": dict(domain_counts),
            "strategy_distribution": dict(strategy_counts),
            "cross_document_examples": cross_doc_count,
            "source_document_distribution": dict(source_doc_counts),
        }

        return stats

    def print_statistics(self):
        """Print formatted dataset statistics."""
        stats = self.get_statistics()

        if not stats:
            print("No data to analyze")
            return

        print(f"\n{'='*60}")
        print(f"GOLDEN DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Cross-document examples: {stats['cross_document_examples']}")

        print(f"\nDifficulty Distribution:")
        for difficulty, count in sorted(stats['difficulty_distribution'].items()):
            pct = (count / stats['total_examples']) * 100
            print(f"  {difficulty.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

        print(f"\nQuery Type Distribution:")
        for qtype, count in sorted(stats['query_type_distribution'].items()):
            pct = (count / stats['total_examples']) * 100
            print(f"  {qtype.capitalize():15s}: {count:3d} ({pct:5.1f}%)")

        print(f"\nDomain Distribution:")
        for domain, count in sorted(stats['domain_distribution'].items()):
            pct = (count / stats['total_examples']) * 100
            print(f"  {domain:20s}: {count:3d} ({pct:5.1f}%)")

        print(f"\nExpected Strategy Distribution:")
        for strategy, count in sorted(stats['strategy_distribution'].items()):
            pct = (count / stats['total_examples']) * 100
            print(f"  {strategy.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

        print(f"\nSource Document Distribution:")
        for doc, count in sorted(stats['source_document_distribution'].items(), key=lambda x: x[1], reverse=True):
            doc_name = doc.split('/')[-1] if '/' in doc else doc
            print(f"  {doc_name[:50]:50s}: {count:3d}")

        print(f"{'='*60}\n")

    def validate_against_corpus(self, retriever) -> Dict:
        """Verify that all chunk IDs in the dataset exist in the corpus."""
        all_chunk_ids = set()

        if hasattr(retriever, 'semantic_retriever') and hasattr(retriever.semantic_retriever, 'vectorstore'):
            try:
                docs = retriever.semantic_retriever.vectorstore.docstore._dict.values()
                all_chunk_ids = {doc.metadata.get('id') for doc in docs if 'id' in doc.metadata}
            except Exception as e:
                print(f"Warning: Could not extract chunk IDs from retriever: {e}")
                return {"error": str(e)}

        missing_chunks = []
        total_chunks_referenced = 0

        for example in self.dataset:
            example_id = example.get('id')
            relevant_doc_ids = example.get('relevant_doc_ids', [])
            total_chunks_referenced += len(relevant_doc_ids)

            for chunk_id in relevant_doc_ids:
                if chunk_id not in all_chunk_ids:
                    missing_chunks.append({
                        'example_id': example_id,
                        'missing_chunk_id': chunk_id
                    })

        results = {
            'total_examples': len(self.dataset),
            'total_chunks_referenced': total_chunks_referenced,
            'total_chunks_in_corpus': len(all_chunk_ids),
            'missing_chunks_count': len(missing_chunks),
            'missing_chunks': missing_chunks[:10],  # Show first 10
            'validation_passed': len(missing_chunks) == 0
        }

        return results


def evaluate_on_golden_dataset(
    graph,
    dataset: List[Dict],
    verbose: bool = True
) -> Dict:
    """
    Run RAG graph on golden dataset and calculate comprehensive metrics.

    Returns: Aggregated metrics and per-example results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"GOLDEN DATASET EVALUATION")
        print(f"{'='*70}")
        print(f"Evaluating {len(dataset)} examples...\n")

    per_example_results = []
    retrieval_metrics_agg = defaultdict(list)
    generation_metrics_agg = defaultdict(list)

    for i, example in enumerate(dataset, 1):
        example_id = example['id']
        question = example['question']

        if verbose:
            print(f"[{i}/{len(dataset)}] Evaluating: {example_id}")
            print(f"  Question: {question[:80]}...")

        state = {
            "user_question": question,
            "baseline_query": question,
            "retrieval_attempts": 0,
            "query_expansions": [],
            "messages": [],
            "retrieved_docs": [],
            "ground_truth_doc_ids": set(example.get('relevant_doc_ids', [])),
            "relevance_grades": example.get('relevance_grades', {}),
        }

        try:
            config = {"configurable": {"thread_id": f"eval-{example_id}"}}
            result = graph.invoke(state, config=config)

            retrieval_metrics = result.get('retrieval_metrics', {})
            groundedness_score = result.get('groundedness_score', 0.0)
            confidence_score = result.get('confidence_score', 0.0)
            has_hallucination = result.get('has_hallucination', False)
            final_answer = result.get('final_answer', '')

            answer_comparison = compare_answers(
                generated=final_answer,
                ground_truth=example['ground_truth_answer']
            )

            answer_relevance = calculate_answer_relevance(
                question=question,
                answer=final_answer
            )

            for metric_name, value in retrieval_metrics.items():
                retrieval_metrics_agg[metric_name].append(value)

            generation_metrics_agg['groundedness'].append(groundedness_score)
            generation_metrics_agg['confidence'].append(confidence_score)
            generation_metrics_agg['has_hallucination'].append(int(has_hallucination))
            generation_metrics_agg['semantic_similarity'].append(answer_comparison.get('semantic_similarity', 0.0))
            generation_metrics_agg['factual_accuracy'].append(answer_comparison.get('factual_accuracy', 0.0))
            generation_metrics_agg['completeness'].append(answer_comparison.get('completeness', 0.0))
            generation_metrics_agg['answer_relevance'].append(answer_relevance.get('relevance_score', 0.0))

            per_example_results.append({
                'example_id': example_id,
                'question': question,
                'difficulty': example.get('difficulty'),
                'query_type': example.get('query_type'),
                'retrieval_metrics': retrieval_metrics,
                'groundedness_score': groundedness_score,
                'confidence_score': confidence_score,
                'has_hallucination': has_hallucination,
                'final_answer': final_answer,
                'ground_truth_answer': example['ground_truth_answer'],
                'semantic_similarity': answer_comparison.get('semantic_similarity', 0.0),
                'factual_accuracy': answer_comparison.get('factual_accuracy', 0.0),
                'completeness': answer_comparison.get('completeness', 0.0),
                'answer_relevance_score': answer_relevance.get('relevance_score', 0.0),
                'is_answer_relevant': answer_relevance.get('is_relevant', False),
                'relevance_category': answer_relevance.get('relevance_category', 'low'),
            })

            if verbose:
                print(f"    Recall@K: {retrieval_metrics.get('recall_at_k', 0):.2%}")
                print(f"    Groundedness: {groundedness_score:.2%}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            per_example_results.append({
                'example_id': example_id,
                'error': str(e)
            })

    avg_retrieval_metrics = {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in retrieval_metrics_agg.items()
    }

    avg_generation_metrics = {
        'avg_groundedness': sum(generation_metrics_agg['groundedness']) / len(generation_metrics_agg['groundedness']) if generation_metrics_agg['groundedness'] else 0.0,
        'avg_confidence': sum(generation_metrics_agg['confidence']) / len(generation_metrics_agg['confidence']) if generation_metrics_agg['confidence'] else 0.0,
        'hallucination_rate': sum(generation_metrics_agg['has_hallucination']) / len(generation_metrics_agg['has_hallucination']) if generation_metrics_agg['has_hallucination'] else 0.0,
        'avg_semantic_similarity': sum(generation_metrics_agg['semantic_similarity']) / len(generation_metrics_agg['semantic_similarity']) if generation_metrics_agg['semantic_similarity'] else 0.0,
        'avg_factual_accuracy': sum(generation_metrics_agg['factual_accuracy']) / len(generation_metrics_agg['factual_accuracy']) if generation_metrics_agg['factual_accuracy'] else 0.0,
        'avg_completeness': sum(generation_metrics_agg['completeness']) / len(generation_metrics_agg['completeness']) if generation_metrics_agg['completeness'] else 0.0,
        'avg_answer_relevance': sum(generation_metrics_agg['answer_relevance']) / len(generation_metrics_agg['answer_relevance']) if generation_metrics_agg['answer_relevance'] else 0.0,
    }

    per_difficulty_breakdown = defaultdict(lambda: defaultdict(list))
    for result in per_example_results:
        if 'error' not in result:
            difficulty = result['difficulty']
            for metric, value in result['retrieval_metrics'].items():
                per_difficulty_breakdown[difficulty][metric].append(value)

    difficulty_metrics = {}
    for difficulty, metrics in per_difficulty_breakdown.items():
        difficulty_metrics[difficulty] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in metrics.items()
        }

    per_query_type_breakdown = defaultdict(lambda: defaultdict(list))
    for result in per_example_results:
        if 'error' not in result:
            query_type = result['query_type']
            for metric, value in result['retrieval_metrics'].items():
                per_query_type_breakdown[query_type][metric].append(value)

    query_type_metrics = {}
    for query_type, metrics in per_query_type_breakdown.items():
        query_type_metrics[query_type] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in metrics.items()
        }

    results = {
        'retrieval_metrics': avg_retrieval_metrics,
        'generation_metrics': avg_generation_metrics,
        'per_difficulty_breakdown': difficulty_metrics,
        'per_query_type_breakdown': query_type_metrics,
        'per_example_results': per_example_results,
        'total_examples': len(dataset),
        'successful_evaluations': len([r for r in per_example_results if 'error' not in r]),
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {results['successful_evaluations']}/{results['total_examples']}")
        print(f"\nRetrieval Metrics:")
        for metric, value in avg_retrieval_metrics.items():
            print(f"  {metric:20s}: {value:.2%}" if value <= 1.0 else f"  {metric:20s}: {value:.4f}")
        print(f"\nGeneration Metrics:")
        for metric, value in avg_generation_metrics.items():
            print(f"  {metric:20s}: {value:.2%}" if 'rate' in metric or value <= 1.0 else f"  {metric:20s}: {value:.4f}")
        print(f"{'='*70}\n")

    return results


class AnswerComparison(BaseModel):
    """
    Structured output schema for comparing generated answers to ground truth.

    Uses semantic field names and descriptions for 95%+ parsing accuracy with GPT-5.
    """
    semantic_similarity: float = Field(
        ge=0.0, le=1.0,
        description="How closely the generated answer matches ground truth meaning (0.0-1.0)"
    )
    factual_accuracy: float = Field(
        ge=0.0, le=1.0,
        description="Correctness of all factual claims in the generated answer (0.0-1.0)"
    )
    completeness: float = Field(
        ge=0.0, le=1.0,
        description="Coverage of all key points from ground truth (0.0-1.0)"
    )
    explanation: str = Field(
        description="Concise reasoning for the scores"
    )


def compare_answers(
    generated: str,
    ground_truth: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict:
    """
    Use LLM with structured output to compare generated answer to ground truth.

    Uses GPT-5-mini with high reasoning effort and Pydantic schema validation
    for 85-90% human agreement and 95%+ parsing accuracy. Prompt optimized for
    GPT-5: concise instructions, XML markup, no CoT scaffolding.

    Hardcoded for consistent, high-quality evaluation across all tier comparisons.
    """
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            reasoning_effort="high"
        )

    # Bind Pydantic schema for structured output (95%+ parsing reliability)
    structured_llm = llm.with_structured_output(AnswerComparison)

    # GPT-5 optimized prompt: concise, XML-structured, direct task specification
    comparison_prompt = f"""Compare the generated answer against the ground truth answer.

<ground_truth>
{ground_truth}
</ground_truth>

<generated_answer>
{generated}
</generated_answer>

Rate on three dimensions (0.0-1.0 scale):
- semantic_similarity: How closely does the generated answer match the ground truth's meaning?
- factual_accuracy: Are all factual claims correct?
- completeness: Does it cover all key points?"""

    try:
        result = structured_llm.invoke(comparison_prompt)
        return result.model_dump()  # Convert Pydantic model to dict
    except Exception as e:
        print(f"Warning: Answer comparison failed: {e}. Using fallback scores.")
        return {
            "semantic_similarity": 0.0,
            "factual_accuracy": 0.0,
            "completeness": 0.0,
            "explanation": f"Evaluation failed: {str(e)}"
        }
