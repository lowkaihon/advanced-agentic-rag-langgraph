"""
RAGAS (Retrieval-Augmented Generation Assessment) Evaluator for RAG Systems.

This module provides integration with the RAGAS evaluation framework to assess
RAG pipeline performance using industry-standard metrics.

RAGAS Metrics:
- Faithfulness: Measures if generated answers contain hallucinations
- Context Recall: Determines if retrieved contexts cover ground truth
- Context Precision: Evaluates if relevant contexts are ranked higher
- Answer Relevancy: Measures how relevant the answer is to the question
"""

import asyncio
from typing import List, Dict, Optional, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    ResponseRelevancy
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.run_config import RunConfig
from collections import defaultdict


class RAGASEvaluator:
    """
    Wrapper for RAGAS evaluation framework.

    Provides methods to:
    - Initialize RAGAS metrics with LLM and embeddings
    - Evaluate single samples
    - Run batch evaluation on datasets
    - Compare RAGAS metrics with custom metrics
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize RAGAS evaluator with LLM and embedding models.

        Args:
            llm_model: OpenAI model name for LLM-based metrics
            temperature: Temperature for LLM evaluation (0.0 for deterministic)
            embedding_model: Optional embedding model name (defaults to OpenAI default)
        """
        # Initialize LLMs and embeddings
        self.evaluator_llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.evaluator_embeddings = OpenAIEmbeddings(
            model=embedding_model if embedding_model else "text-embedding-3-small"
        )

        # Wrap for RAGAS compatibility
        self.llm_wrapper = LangchainLLMWrapper(self.evaluator_llm)
        self.embeddings_wrapper = LangchainEmbeddingsWrapper(self.evaluator_embeddings)

        # Initialize metrics
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> List:
        """
        Initialize and configure RAGAS metrics.

        Returns:
            List of configured RAGAS metric instances
        """
        metrics = [
            Faithfulness(),
            ContextRecall(),
            ContextPrecision(),
            ResponseRelevancy()
        ]

        # Configure each metric with LLM and embeddings
        run_config = RunConfig()

        for metric in metrics:
            if hasattr(metric, 'llm'):
                metric.llm = self.llm_wrapper
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.embeddings_wrapper

            # Initialize metric
            metric.init(run_config)

        return metrics

    def prepare_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> SingleTurnSample:
        """
        Prepare a single sample in RAGAS format.

        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer for Context Recall

        Returns:
            SingleTurnSample ready for RAGAS evaluation
        """
        sample_data = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts
        }

        # Add ground truth if available (needed for Context Recall)
        if ground_truth:
            sample_data["reference"] = ground_truth

        return SingleTurnSample(**sample_data)

    async def evaluate_sample(
        self,
        sample: SingleTurnSample,
        metrics: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single sample with RAGAS metrics.

        Args:
            sample: SingleTurnSample to evaluate
            metrics: Optional list of metrics (uses all if not specified)

        Returns:
            Dictionary mapping metric names to scores
        """
        if metrics is None:
            metrics = self.metrics

        scores = {}

        for metric in metrics:
            try:
                # Run async evaluation
                score = await metric.single_turn_ascore(sample)
                scores[metric.name] = score
            except Exception as e:
                print(f"Warning: Failed to compute {metric.name}: {e}")
                scores[metric.name] = None

        return scores

    def evaluate_sample_sync(
        self,
        sample: SingleTurnSample,
        metrics: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Synchronous wrapper for evaluate_sample.

        Args:
            sample: SingleTurnSample to evaluate
            metrics: Optional list of metrics

        Returns:
            Dictionary mapping metric names to scores
        """
        return asyncio.run(self.evaluate_sample(sample, metrics))

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run batch evaluation on an EvaluationDataset.

        Args:
            dataset: EvaluationDataset containing samples
            metrics: Optional list of metrics (uses all if not specified)

        Returns:
            Dictionary with evaluation results and statistics
        """
        if metrics is None:
            metrics = self.metrics

        # Run RAGAS batch evaluation
        results = await evaluate(
            dataset,
            metrics=metrics,
            llm=self.llm_wrapper,
            embeddings=self.embeddings_wrapper
        )

        return results

    def evaluate_dataset_sync(
        self,
        dataset: EvaluationDataset,
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_dataset.

        Args:
            dataset: EvaluationDataset to evaluate
            metrics: Optional list of metrics

        Returns:
            Dictionary with evaluation results
        """
        return asyncio.run(self.evaluate_dataset(dataset, metrics))


def prepare_ragas_dataset_from_golden(
    golden_dataset: List[Dict],
    graph_results: List[Dict]
) -> EvaluationDataset:
    """
    Convert golden dataset and graph results into RAGAS EvaluationDataset.

    Args:
        golden_dataset: List of golden dataset examples with ground truth
        graph_results: List of graph execution results matching golden examples

    Returns:
        EvaluationDataset ready for RAGAS evaluation
    """
    samples = []

    for golden_example, graph_result in zip(golden_dataset, graph_results):
        # Extract data from golden example
        question = golden_example['question']
        ground_truth = golden_example['ground_truth_answer']

        # Extract data from graph result
        answer = graph_result.get('final_answer', '')

        # Get retrieved contexts
        retrieved_docs = graph_result.get('retrieved_docs', [])
        contexts = [doc if isinstance(doc, str) else doc.page_content
                   for doc in retrieved_docs]

        # Create sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth
        )

        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_ragas_evaluation_on_golden(
    golden_dataset: List[Dict],
    graph,
    evaluator: Optional[RAGASEvaluator] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete RAGAS evaluation on golden dataset.

    Steps:
    1. Execute graph on each golden example
    2. Prepare RAGAS dataset from results
    3. Run RAGAS batch evaluation
    4. Return comprehensive metrics

    Args:
        golden_dataset: List of golden dataset examples
        graph: Compiled LangGraph instance
        evaluator: Optional RAGASEvaluator instance (creates default if None)
        verbose: Whether to print progress

    Returns:
        Dictionary with RAGAS metrics and per-example results
    """
    if evaluator is None:
        evaluator = RAGASEvaluator()

    if verbose:
        print(f"\n{'='*70}")
        print(f"RAGAS EVALUATION ON GOLDEN DATASET")
        print(f"{'='*70}")
        print(f"Evaluating {len(golden_dataset)} examples...\n")

    # Execute graph on all examples
    graph_results = []

    for i, example in enumerate(golden_dataset, 1):
        example_id = example['id']
        question = example['question']

        if verbose:
            print(f"[{i}/{len(golden_dataset)}] Running: {example_id}")

        # Prepare state
        state = {
            "question": question,
            "original_query": question,
            "conversation_history": [],
            "retrieval_attempts": 0,
            "query_expansions": [],
            "messages": [],
            "retrieved_docs": [],
            "ground_truth_doc_ids": set(example.get('relevant_doc_ids', [])),
            "relevance_grades": example.get('relevance_grades', {}),
        }

        try:
            # Execute graph
            config = {"configurable": {"thread_id": f"ragas-eval-{example_id}"}}
            result = graph.invoke(state, config=config)
            graph_results.append(result)

            if verbose:
                print(f"  Status: Success")

        except Exception as e:
            if verbose:
                print(f"  Status: ERROR - {str(e)}")
            graph_results.append({"error": str(e)})

    # Prepare RAGAS dataset
    if verbose:
        print(f"\nPreparing RAGAS dataset...")

    ragas_dataset = prepare_ragas_dataset_from_golden(
        golden_dataset,
        graph_results
    )

    # Run RAGAS evaluation
    if verbose:
        print(f"Running RAGAS metrics evaluation...")

    ragas_results = evaluator.evaluate_dataset_sync(ragas_dataset)

    # Calculate aggregate statistics
    if verbose:
        print(f"\n{'='*70}")
        print(f"RAGAS EVALUATION RESULTS")
        print(f"{'='*70}")

        # Print metric scores
        if hasattr(ragas_results, 'scores'):
            for metric_name in ragas_results.scores.columns:
                scores = ragas_results.scores[metric_name].dropna()
                if len(scores) > 0:
                    avg_score = scores.mean()
                    print(f"{metric_name:25s}: {avg_score:.4f}")

        print(f"{'='*70}\n")

    return {
        'ragas_results': ragas_results,
        'graph_results': graph_results,
        'total_examples': len(golden_dataset),
        'successful_evaluations': len([r for r in graph_results if 'error' not in r])
    }


def compare_ragas_with_custom_metrics(
    ragas_results: Dict,
    custom_results: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare RAGAS metrics with custom evaluation metrics.

    Analyzes correlation between:
    - Faithfulness (RAGAS) vs. Groundedness (custom)
    - Context Precision (RAGAS) vs. Retrieval Quality (custom)
    - Response Relevancy (RAGAS) vs. Answer Sufficiency (custom)

    Args:
        ragas_results: Results from RAGAS evaluation
        custom_results: Results from custom evaluation pipeline
        verbose: Whether to print comparison

    Returns:
        Dictionary with comparison analysis
    """
    comparison = {
        'ragas_metrics': {},
        'custom_metrics': {},
        'correlations': {},
        'insights': []
    }

    # Extract RAGAS metric averages
    if hasattr(ragas_results, 'scores'):
        for metric_name in ragas_results.scores.columns:
            scores = ragas_results.scores[metric_name].dropna()
            if len(scores) > 0:
                comparison['ragas_metrics'][metric_name] = float(scores.mean())

    # Extract custom metric averages
    if 'generation_metrics' in custom_results:
        comparison['custom_metrics']['groundedness'] = custom_results['generation_metrics'].get('avg_groundedness', 0.0)
        comparison['custom_metrics']['confidence'] = custom_results['generation_metrics'].get('avg_confidence', 0.0)
        comparison['custom_metrics']['hallucination_rate'] = custom_results['generation_metrics'].get('hallucination_rate', 0.0)

    if 'retrieval_metrics' in custom_results:
        comparison['custom_metrics']['recall_at_k'] = custom_results['retrieval_metrics'].get('recall_at_k', 0.0)
        comparison['custom_metrics']['precision_at_k'] = custom_results['retrieval_metrics'].get('precision_at_k', 0.0)

    # Analyze correlations and generate insights
    if 'faithfulness' in comparison['ragas_metrics'] and 'groundedness' in comparison['custom_metrics']:
        faithfulness = comparison['ragas_metrics']['faithfulness']
        groundedness = comparison['custom_metrics']['groundedness']

        diff = abs(faithfulness - groundedness)
        comparison['correlations']['faithfulness_vs_groundedness'] = {
            'ragas_faithfulness': faithfulness,
            'custom_groundedness': groundedness,
            'difference': diff,
            'correlation_strength': 'high' if diff < 0.1 else 'moderate' if diff < 0.2 else 'low'
        }

        if diff < 0.1:
            comparison['insights'].append(
                f"Strong alignment between RAGAS Faithfulness ({faithfulness:.2%}) "
                f"and custom Groundedness ({groundedness:.2%})"
            )
        else:
            comparison['insights'].append(
                f"Divergence between RAGAS Faithfulness ({faithfulness:.2%}) "
                f"and custom Groundedness ({groundedness:.2%}) - investigate causes"
            )

    if verbose:
        print(f"\n{'='*70}")
        print(f"RAGAS vs CUSTOM METRICS COMPARISON")
        print(f"{'='*70}")

        print(f"\nRAGAS Metrics:")
        for metric, value in comparison['ragas_metrics'].items():
            print(f"  {metric:25s}: {value:.4f}")

        print(f"\nCustom Metrics:")
        for metric, value in comparison['custom_metrics'].items():
            print(f"  {metric:25s}: {value:.4f}")

        if comparison['insights']:
            print(f"\nKey Insights:")
            for insight in comparison['insights']:
                print(f"  - {insight}")

        print(f"{'='*70}\n")

    return comparison
