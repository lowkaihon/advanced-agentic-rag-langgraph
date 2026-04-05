"""Model tier configuration for LLM selection across RAG pipeline tasks."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import os


class ModelTier(str, Enum):
    """Available model tier configurations."""
    BUDGET = "budget"
    BALANCED = "balanced"
    PREMIUM = "premium"


@dataclass
class ModelSpec:
    """LLM model configuration with tier-specific optimizations."""
    name: str
    temperature: float
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    few_shot_count: int = 0


@dataclass
class TierConfig:
    """Complete tier configuration mapping RAG tasks to model specs."""
    conversational_rewrite: ModelSpec
    expansion_decision: ModelSpec
    query_expansion: ModelSpec
    strategy_selection: ModelSpec
    retrieval_quality_eval: ModelSpec
    query_rewriting: ModelSpec
    strategy_optimization: ModelSpec
    llm_reranking: ModelSpec
    answer_generation: ModelSpec
    hhem_claim_decomposition: ModelSpec
    answer_quality_eval: ModelSpec
    ragas_evaluation: ModelSpec
    # Multi-agent RAG tasks
    complexity_classification: ModelSpec
    query_decomposition: ModelSpec
    multi_agent_merge_reranking: ModelSpec


# ========== BUDGET TIER: ALL GPT-4o-mini ==========
# Cost: $1,200/day | Quality: 70-75%
# Strategy: Showcase RAG architecture value with cheapest model
# Optimizations: Few-shot examples, explicit CoT scaffolding, medium verbosity

BUDGET_TIER = TierConfig(
    conversational_rewrite=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Simple task, no examples needed
    ),
    expansion_decision=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Binary classification
    ),
    query_expansion=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Few-shot improves variant quality
    ),
    strategy_selection=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Well-defined enumeration
    ),
    retrieval_quality_eval=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Multi-criterion evaluation benefits from examples
    ),
    query_rewriting=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Retry path, needs guidance
    ),
    strategy_optimization=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Rare path, keep simple
    ),
    llm_reranking=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Nuanced scoring benefits from examples
    ),
    answer_generation=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Explicit structure in prompt
    ),
    hhem_claim_decomposition=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=3  # Critical task, needs reasoning traces
    ),
    answer_quality_eval=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Strict rubric with examples
    ),
    ragas_evaluation=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Offline, use cheapest
    ),
    # Multi-agent RAG tasks
    complexity_classification=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,  # Deterministic classification
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Binary decision, no examples needed
    ),
    query_decomposition=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,  # Consistent decomposition
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Few-shot improves decomposition quality
    ),
    multi_agent_merge_reranking=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,  # Deterministic selection
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Coverage selection is well-defined
    ),
)


# ========== BALANCED TIER: HYBRID GPT-4o-mini + GPT-5-mini ==========
# Cost: $1,800/day | Quality: 78-80%
# Strategy: GPT-4o-mini for sequential (latency), GPT-5-mini for async (quality)
# Optimizations: Mixed reasoning_effort, reduced few-shot for GPT-5-mini tasks

BALANCED_TIER = TierConfig(
    # Sequential tasks: GPT-4o-mini (latency-sensitive)
    conversational_rewrite=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0
    ),
    expansion_decision=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0
    ),
    query_expansion=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2
    ),
    strategy_selection=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0
    ),

    # Async tasks: GPT-5-mini (quality-critical)
    retrieval_quality_eval=ModelSpec(
        name="gpt-5-mini",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0  # GPT-5 doesn't need examples
    ),
    query_rewriting=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Retry path, keep fast
    ),
    strategy_optimization=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0
    ),
    llm_reranking=ModelSpec(
        name="gpt-5-mini",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0
    ),
    answer_generation=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.7,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Sequential, latency-sensitive
    ),
    hhem_claim_decomposition=ModelSpec(
        name="gpt-5-mini",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0  # GPT-5 excels at hallucination detection
    ),
    answer_quality_eval=ModelSpec(
        name="gpt-5-mini",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0
    ),
    ragas_evaluation=ModelSpec(
        name="gpt-4o-mini",
        temperature=0.0,
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Offline, use cheaper
    ),
    # Multi-agent RAG tasks
    complexity_classification=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,  # Deterministic classification
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=0  # Binary decision
    ),
    query_decomposition=ModelSpec(
        name="gpt-4o-mini",
        temperature=0,  # Consistent decomposition
        reasoning_effort=None,
        verbosity="medium",
        few_shot_count=2  # Keep latency-sensitive
    ),
    multi_agent_merge_reranking=ModelSpec(
        name="gpt-5-mini",
        temperature=0,  # Deterministic selection
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0  # GPT-5 doesn't need examples
    ),
)


# ========== PREMIUM TIER: GPT-5.1 + GPT-5-mini + GPT-5-nano ==========
# Cost: $12,060/day | Quality: 88-92%
# Strategy: Best models for all tasks, optimized reasoning_effort per task
# Optimizations: High reasoning for evaluation, low for generation, no few-shot

PREMIUM_TIER = TierConfig(
    conversational_rewrite=ModelSpec(
        name="gpt-5-mini",
        temperature=0,
        reasoning_effort="low",
        verbosity="low",
        few_shot_count=0
    ),
    expansion_decision=ModelSpec(
        name="gpt-5-nano",
        temperature=0,
        reasoning_effort="minimal",
        verbosity="low",
        few_shot_count=0  # Binary classification, lightweight
    ),
    query_expansion=ModelSpec(
        name="gpt-5-mini",
        temperature=0.7,
        reasoning_effort="low",
        verbosity="low",
        few_shot_count=0
    ),
    strategy_selection=ModelSpec(
        name="gpt-5-mini",
        temperature=0,
        reasoning_effort="low",
        verbosity="low",
        few_shot_count=0
    ),
    retrieval_quality_eval=ModelSpec(
        name="gpt-5.1",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0
    ),
    query_rewriting=ModelSpec(
        name="gpt-5-mini",
        temperature=0.7,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0
    ),
    strategy_optimization=ModelSpec(
        name="gpt-5-nano",
        temperature=0.7,
        reasoning_effort="low",
        verbosity="low",
        few_shot_count=0  # Lightweight, rare path
    ),
    llm_reranking=ModelSpec(
        name="gpt-5.1",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0
    ),
    answer_generation=ModelSpec(
        name="gpt-5.1",
        temperature=0.7,
        reasoning_effort="low",  # NOT high (avoid latency)
        verbosity="low",  # Constrain verbose outputs
        few_shot_count=0
    ),
    hhem_claim_decomposition=ModelSpec(
        name="gpt-5.1",
        temperature=0,
        reasoning_effort="high",  # Critical task, leverage deep reasoning
        verbosity="low",
        few_shot_count=0  # Trust internal reasoning
    ),
    answer_quality_eval=ModelSpec(
        name="gpt-5.1",
        temperature=0,
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0
    ),
    ragas_evaluation=ModelSpec(
        name="gpt-5-mini",
        temperature=0.0,
        reasoning_effort=None,
        verbosity="low",
        few_shot_count=0  # Offline, cheaper than GPT-5.1
    ),
    # Multi-agent RAG tasks
    complexity_classification=ModelSpec(
        name="gpt-5-nano",
        temperature=0,  # Deterministic classification
        reasoning_effort="minimal",
        verbosity="low",
        few_shot_count=0  # Lightweight binary decision
    ),
    query_decomposition=ModelSpec(
        name="gpt-5-mini",
        temperature=0,  # Consistent decomposition
        reasoning_effort="low",
        verbosity="low",
        few_shot_count=0  # GPT-5 doesn't need examples
    ),
    multi_agent_merge_reranking=ModelSpec(
        name="gpt-5.1",
        temperature=0,  # Deterministic selection
        reasoning_effort="medium",
        verbosity="low",
        few_shot_count=0  # GPT-5 excels at coverage reasoning
    ),
)


# ========== TIER REGISTRY ==========

TIER_CONFIGS = {
    ModelTier.BUDGET: BUDGET_TIER,
    ModelTier.BALANCED: BALANCED_TIER,
    ModelTier.PREMIUM: PREMIUM_TIER,
}


# ========== PUBLIC API ==========

def get_current_tier() -> ModelTier:
    """Get active tier from MODEL_TIER env var (defaults to BUDGET)."""
    tier_str = os.getenv("MODEL_TIER", "budget").lower()
    try:
        return ModelTier(tier_str)
    except ValueError:
        print(f"Warning: Invalid MODEL_TIER '{tier_str}', defaulting to 'budget'")
        return ModelTier.BUDGET


def get_model_for_task(task_name: str) -> ModelSpec:
    """Get model specification for a task in the current tier."""
    tier = get_current_tier()
    config = TIER_CONFIGS[tier]

    try:
        return getattr(config, task_name)
    except AttributeError:
        raise AttributeError(
            f"Invalid task name '{task_name}'. Valid tasks: "
            f"{', '.join([attr for attr in dir(config) if not attr.startswith('_')])}"
        )


def reset_llm_cache():
    """Placeholder for resetting cached LLM instances when switching tiers."""
    # TODO: Implement if modules start caching LLM instances at module level
    pass


# ========== TIER COMPARISON METADATA ==========

TIER_METADATA = {
    ModelTier.BUDGET: {
        "cost_per_day": 1200,
        "expected_quality": "70-75%",
        "description": "All GPT-4o-mini - Showcase RAG architecture value with cheapest model",
        "optimizations": [
            "Few-shot examples (2-3 per task)",
            "Explicit CoT scaffolding",
            "Medium verbosity"
        ]
    },
    ModelTier.BALANCED: {
        "cost_per_day": 1800,
        "expected_quality": "78-80%",
        "description": "Hybrid GPT-4o-mini (sequential) + GPT-5-mini (async)",
        "optimizations": [
            "Strategic model allocation",
            "Mixed reasoning_effort",
            "Reduced few-shot for GPT-5 tasks"
        ]
    },
    ModelTier.PREMIUM: {
        "cost_per_day": 12060,
        "expected_quality": "88-92%",
        "description": "GPT-5.1 + GPT-5-mini + GPT-5-nano - Maximum quality",
        "optimizations": [
            "High reasoning_effort for evaluation",
            "Low reasoning_effort for generation (latency)",
            "No few-shot examples (GPT-5 doesn't need them)",
            "Low verbosity (constrain GPT-5 verbose outputs)"
        ]
    }
}
