"""
Golden Dataset Management and Evaluation for RAG Systems.

This module provides utilities for managing golden datasets and conducting
offline evaluation of RAG pipelines using ground truth data.
"""

import json
import os
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


class GoldenDatasetManager:
    """
    Manage and validate golden datasets for RAG evaluation.

    Features:
    - Load/save golden datasets with validation
    - Filter examples by difficulty, query type, domain
    - Validate dataset structure and completeness
    - Verify chunk IDs against corpus
    - Generate dataset statistics
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the GoldenDatasetManager.

        Args:
            dataset_path: Path to the golden dataset JSON file
        """
        self.dataset_path = dataset_path
        self.dataset = []

        if os.path.exists(dataset_path):
            self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Dict]:
        """
        Load and validate golden dataset from JSON file.

        Returns:
            List of validated example dictionaries

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset has invalid structure
        """
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
        """
        Save dataset to JSON file with validation.

        Args:
            dataset: List of example dictionaries to save
        """
        # Validate all examples before saving
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

        Checks:
        - Required fields present
        - Relevance grades match doc_ids
        - Difficulty is valid value
        - Query type is valid
        - Ground truth answer is reasonable length

        Args:
            example: Example dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Required fields
        required_fields = [
            "id", "question", "ground_truth_answer", "relevant_doc_ids",
            "source_document", "difficulty", "query_type", "domain", "expected_strategy"
        ]

        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")

        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        if example.get("difficulty") not in valid_difficulties:
            errors.append(f"Invalid difficulty: {example.get('difficulty')}")

        # Validate query type
        valid_query_types = ["factual", "conceptual", "procedural", "comparative"]
        if example.get("query_type") not in valid_query_types:
            errors.append(f"Invalid query_type: {example.get('query_type')}")

        # Validate expected strategy
        valid_strategies = ["semantic", "keyword", "hybrid"]
        if example.get("expected_strategy") not in valid_strategies:
            errors.append(f"Invalid expected_strategy: {example.get('expected_strategy')}")

        # Validate ground truth answer length
        answer = example.get("ground_truth_answer", "")
        if len(answer) < 50:
            errors.append("Ground truth answer too short (< 50 chars)")
        if len(answer) > 2000:
            errors.append("Ground truth answer too long (> 2000 chars)")

        # Validate relevant_doc_ids is a list
        if not isinstance(example.get("relevant_doc_ids"), list):
            errors.append("relevant_doc_ids must be a list")

        # Validate relevance_grades if present
        if "relevance_grades" in example:
            relevance_grades = example["relevance_grades"]
            doc_ids = set(example.get("relevant_doc_ids", []))

            # Check that graded doc_ids are subset of relevant_doc_ids
            graded_ids = set(relevance_grades.keys())
            if not graded_ids.issubset(doc_ids):
                errors.append("relevance_grades contains doc_ids not in relevant_doc_ids")

            # Check that grades are 0-3
            for doc_id, grade in relevance_grades.items():
                if not isinstance(grade, int) or grade not in [0, 1, 2, 3]:
                    errors.append(f"Invalid relevance grade for {doc_id}: {grade} (must be 0-3)")

        return (len(errors) == 0, errors)

    def add_example(self, example: Dict) -> bool:
        """
        Add new example to dataset with validation.

        Args:
            example: Example dictionary to add

        Returns:
            True if added successfully, False otherwise
        """
        is_valid, errors = self.validate_example(example)
        if not is_valid:
            print(f"Cannot add invalid example: {', '.join(errors)}")
            return False

        self.dataset.append(example)
        return True

    def get_by_difficulty(self, difficulty: str) -> List[Dict]:
        """
        Filter examples by difficulty level.

        Args:
            difficulty: "easy", "medium", or "hard"

        Returns:
            List of examples matching difficulty
        """
        return [ex for ex in self.dataset if ex.get("difficulty") == difficulty]

    def get_by_query_type(self, query_type: str) -> List[Dict]:
        """
        Filter examples by query type.

        Args:
            query_type: "factual", "conceptual", "procedural", or "comparative"

        Returns:
            List of examples matching query type
        """
        return [ex for ex in self.dataset if ex.get("query_type") == query_type]

    def get_by_domain(self, domain: str) -> List[Dict]:
        """
        Filter examples by domain.

        Args:
            domain: Domain name (e.g., "nlp", "computer_vision", "generative_models")

        Returns:
            List of examples matching domain
        """
        return [ex for ex in self.dataset if ex.get("domain") == domain]

    def get_cross_document_examples(self) -> List[Dict]:
        """
        Get examples that involve multiple source documents.

        Returns:
            List of cross-document examples
        """
        return [
            ex for ex in self.dataset
            if isinstance(ex.get("source_document"), list) and len(ex["source_document"]) > 1
        ]

    def get_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.

        Returns:
            Dictionary with dataset composition stats
        """
        if not self.dataset:
            return {}

        # Count by difficulty
        difficulty_counts = defaultdict(int)
        for ex in self.dataset:
            difficulty_counts[ex.get("difficulty", "unknown")] += 1

        # Count by query type
        query_type_counts = defaultdict(int)
        for ex in self.dataset:
            query_type_counts[ex.get("query_type", "unknown")] += 1

        # Count by domain
        domain_counts = defaultdict(int)
        for ex in self.dataset:
            domain_counts[ex.get("domain", "unknown")] += 1

        # Count by expected strategy
        strategy_counts = defaultdict(int)
        for ex in self.dataset:
            strategy_counts[ex.get("expected_strategy", "unknown")] += 1

        # Cross-document examples
        cross_doc_count = len(self.get_cross_document_examples())

        # Source document distribution
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
        """
        Verify that all chunk IDs in the dataset exist in the corpus.

        Args:
            retriever: HybridRetriever instance with loaded documents

        Returns:
            Dictionary with validation results
        """
        all_chunk_ids = set()

        # Get all chunk IDs from retriever's vector store
        # Note: This assumes the retriever exposes its document store
        if hasattr(retriever, 'semantic_retriever') and hasattr(retriever.semantic_retriever, 'vectorstore'):
            # Try to get documents from FAISS store
            try:
                # This is implementation-specific and may need adjustment
                docs = retriever.semantic_retriever.vectorstore.docstore._dict.values()
                all_chunk_ids = {doc.metadata.get('id') for doc in docs if 'id' in doc.metadata}
            except Exception as e:
                print(f"Warning: Could not extract chunk IDs from retriever: {e}")
                return {"error": str(e)}

        # Check each example's chunk IDs
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

    For each example:
    1. Inject ground_truth_doc_ids and relevance_grades into state
    2. Run graph with question
    3. Collect retrieval_metrics from result
    4. Compare generated answer to ground_truth_answer
    5. Track groundedness scores

    Args:
        graph: Compiled LangGraph instance
        dataset: List of golden dataset examples
        verbose: Whether to print progress

    Returns:
        Dictionary with aggregated metrics and per-example results
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

        # Prepare state with ground truth
        state = {
            "question": question,
            "original_query": question,
            "conversation_history": [],
            "retrieval_attempts": 0,
            "query_expansions": [],
            "messages": [],
            "retrieved_docs": [],
            # Inject ground truth for evaluation
            "ground_truth_doc_ids": set(example.get('relevant_doc_ids', [])),
            "relevance_grades": example.get('relevance_grades', {}),
        }

        try:
            # Run the graph
            config = {"configurable": {"thread_id": f"eval-{example_id}"}}
            result = graph.invoke(state, config=config)

            # Extract metrics
            retrieval_metrics = result.get('retrieval_metrics', {})
            groundedness_score = result.get('groundedness_score', 0.0)
            confidence_score = result.get('confidence_score', 0.0)
            has_hallucination = result.get('has_hallucination', False)
            final_answer = result.get('final_answer', '')

            # Aggregate retrieval metrics
            for metric_name, value in retrieval_metrics.items():
                retrieval_metrics_agg[metric_name].append(value)

            # Aggregate generation metrics
            generation_metrics_agg['groundedness'].append(groundedness_score)
            generation_metrics_agg['confidence'].append(confidence_score)
            generation_metrics_agg['has_hallucination'].append(int(has_hallucination))

            # Store per-example result
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
            })

            if verbose:
                print(f"    Recall@5: {retrieval_metrics.get('recall_at_k', 0):.2%}")
                print(f"    Groundedness: {groundedness_score:.2%}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            per_example_results.append({
                'example_id': example_id,
                'error': str(e)
            })

    # Calculate aggregate metrics
    avg_retrieval_metrics = {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in retrieval_metrics_agg.items()
    }

    avg_generation_metrics = {
        'avg_groundedness': sum(generation_metrics_agg['groundedness']) / len(generation_metrics_agg['groundedness']) if generation_metrics_agg['groundedness'] else 0.0,
        'avg_confidence': sum(generation_metrics_agg['confidence']) / len(generation_metrics_agg['confidence']) if generation_metrics_agg['confidence'] else 0.0,
        'hallucination_rate': sum(generation_metrics_agg['has_hallucination']) / len(generation_metrics_agg['has_hallucination']) if generation_metrics_agg['has_hallucination'] else 0.0,
    }

    # Breakdown by difficulty
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

    # Breakdown by query type
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


def compare_answers(
    generated: str,
    ground_truth: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict:
    """
    Use LLM to compare generated answer to ground truth.

    Args:
        generated: Generated answer from RAG system
        ground_truth: Ground truth answer from dataset
        llm: Optional LLM instance (creates default if not provided)

    Returns:
        Dictionary with comparison scores and reasoning
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    comparison_prompt = f"""Compare the generated answer to the ground truth answer.

Ground Truth Answer:
{ground_truth}

Generated Answer:
{generated}

Evaluate the generated answer on these dimensions (0.0-1.0 scale):
1. Semantic Similarity: How similar is the meaning?
2. Factual Accuracy: Are the facts correct?
3. Completeness: Does it cover all key points?

Respond in JSON format:
{{
    "semantic_similarity": 0.0-1.0,
    "factual_accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    response = llm.invoke(comparison_prompt)
    content = response.content

    # Extract JSON
    import re
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            evaluation = json.loads(json_match.group())
            return evaluation
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "semantic_similarity": 0.0,
        "factual_accuracy": 0.0,
        "completeness": 0.0,
        "reasoning": "Failed to parse LLM response"
    }
