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
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
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

    Provides: Initialize metrics, evaluate single samples, batch evaluation, metric comparison.
    """

    def __init__(
        self,
        llm_model: str = None,
        temperature: float = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize RAGAS evaluator with tier-based model configuration.

        Args:
            llm_model: LLM for evaluation (None = use tier config)
            temperature: Sampling temperature (None = use tier config)
            embedding_model: Embedding model (None = use default text-embedding-3-small)
        """
        spec = get_model_for_task("ragas_evaluation")
        llm_model = llm_model or spec.name
        temperature = temperature if temperature is not None else spec.temperature

        self.evaluator_llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity
        )
        self.evaluator_embeddings = OpenAIEmbeddings(
            model=embedding_model if embedding_model else "text-embedding-3-small"
        )

        self.llm_wrapper = LangchainLLMWrapper(self.evaluator_llm)
        self.embeddings_wrapper = LangchainEmbeddingsWrapper(self.evaluator_embeddings)

        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> List:
        """Initialize and configure RAGAS metrics."""
        metrics = [
            Faithfulness(),
            ContextRecall(),
            ContextPrecision(),
            ResponseRelevancy()
        ]

        run_config = RunConfig()

        for metric in metrics:
            if hasattr(metric, 'llm'):
                metric.llm = self.llm_wrapper
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.embeddings_wrapper

            metric.init(run_config)

        return metrics

    def prepare_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> SingleTurnSample:
        """Prepare a single sample in RAGAS format."""
        sample_data = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts
        }

        if ground_truth:
            sample_data["reference"] = ground_truth

        return SingleTurnSample(**sample_data)

    async def evaluate_sample(
        self,
        sample: SingleTurnSample,
        metrics: Optional[List] = None
    ) -> Dict[str, float]:
        """Evaluate a single sample with RAGAS metrics."""
        if metrics is None:
            metrics = self.metrics

        scores = {}

        for metric in metrics:
            try:
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
        """Synchronous wrapper for evaluate_sample."""
        return asyncio.run(self.evaluate_sample(sample, metrics))

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """Run batch evaluation on an EvaluationDataset."""
        if metrics is None:
            metrics = self.metrics

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
        """Synchronous wrapper for evaluate_dataset."""
        return asyncio.run(self.evaluate_dataset(dataset, metrics))


def prepare_ragas_dataset_from_golden(
    golden_dataset: List[Dict],
    graph_results: List[Dict]
) -> EvaluationDataset:
    """Convert golden dataset and graph results into RAGAS EvaluationDataset."""
    samples = []

    for golden_example, graph_result in zip(golden_dataset, graph_results):
        question = golden_example['question']
        ground_truth = golden_example['ground_truth_answer']

        answer = graph_result.get('final_answer', '')

        retrieved_docs = graph_result.get('retrieved_docs', [])
        contexts = [doc if isinstance(doc, str) else doc.page_content
                   for doc in retrieved_docs]

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

    Returns: RAGAS metrics and per-example results
    """
    if evaluator is None:
        evaluator = RAGASEvaluator()

    if verbose:
        print(f"\n{'='*70}")
        print(f"RAGAS EVALUATION ON GOLDEN DATASET")
        print(f"{'='*70}")
        print(f"Evaluating {len(golden_dataset)} examples...\n")

    graph_results = []

    for i, example in enumerate(golden_dataset, 1):
        example_id = example['id']
        question = example['question']

        if verbose:
            print(f"[{i}/{len(golden_dataset)}] Running: {example_id}")

        state = {
            "question": question,
            "original_query": question,
            "retrieval_attempts": 0,
            "query_expansions": [],
            "messages": [],
            "retrieved_docs": [],
            "ground_truth_doc_ids": set(example.get('relevant_doc_ids', [])),
            "relevance_grades": example.get('relevance_grades', {}),
        }

        try:
            config = {"configurable": {"thread_id": f"ragas-eval-{example_id}"}}
            result = graph.invoke(state, config=config)
            graph_results.append(result)

            if verbose:
                print(f"  Status: Success")

        except Exception as e:
            if verbose:
                print(f"  Status: ERROR - {str(e)}")
            graph_results.append({"error": str(e)})

    if verbose:
        print(f"\nPreparing RAGAS dataset...")

    ragas_dataset = prepare_ragas_dataset_from_golden(
        golden_dataset,
        graph_results
    )

    if verbose:
        print(f"Running RAGAS metrics evaluation...")

    ragas_results = evaluator.evaluate_dataset_sync(ragas_dataset)

    if verbose:
        print(f"\n{'='*70}")
        print(f"RAGAS EVALUATION RESULTS")
        print(f"{'='*70}")

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

    Analyzes: Faithfulness vs Groundedness, Context Precision vs Retrieval Quality,
    Response Relevancy vs Answer Sufficiency.
    """
    comparison = {
        'ragas_metrics': {},
        'custom_metrics': {},
        'correlations': {},
        'insights': []
    }

    if hasattr(ragas_results, 'scores'):
        for metric_name in ragas_results.scores.columns:
            scores = ragas_results.scores[metric_name].dropna()
            if len(scores) > 0:
                comparison['ragas_metrics'][metric_name] = float(scores.mean())

    if 'generation_metrics' in custom_results:
        comparison['custom_metrics']['groundedness'] = custom_results['generation_metrics'].get('avg_groundedness', 0.0)
        comparison['custom_metrics']['confidence'] = custom_results['generation_metrics'].get('avg_confidence', 0.0)
        comparison['custom_metrics']['hallucination_rate'] = custom_results['generation_metrics'].get('hallucination_rate', 0.0)

    if 'retrieval_metrics' in custom_results:
        comparison['custom_metrics']['recall_at_k'] = custom_results['retrieval_metrics'].get('recall_at_k', 0.0)
        comparison['custom_metrics']['precision_at_k'] = custom_results['retrieval_metrics'].get('precision_at_k', 0.0)

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
