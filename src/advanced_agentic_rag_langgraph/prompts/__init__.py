"""
Prompt template system with model-specific variant support.

Research-backed optimizations:
- GPT-4o/GPT-4o-mini: Few-shot examples, explicit scaffolding, detailed rubrics
- GPT-5 family: Concise instructions, no examples, trust internal reasoning

Usage:
    from advanced_agentic_rag_langgraph.prompts import get_prompt

    # Automatically selects correct variant based on MODEL_TIER
    prompt = get_prompt("hhem_claim_decomposition", answer="...")
"""

from typing import Optional
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


def get_prompt(task_name: str, **kwargs) -> str:
    """
    Load prompt template with model-specific variant support.

    Resolution logic:
    1. Get model spec for task from tier configuration
    2. Detect model family (GPT-5 vs base/GPT-4o)
    3. Load appropriate variant (GPT5_PROMPT vs BASE_PROMPT)
    4. Format with kwargs if provided

    Args:
        task_name: Task identifier matching prompt module name
                  (e.g., "hhem_claim_decomposition", "answer_quality_eval")
        **kwargs: Template variables for f-string formatting

    Returns:
        Formatted prompt string

    Example:
        >>> # Premium tier (GPT-5.1) loads GPT5_PROMPT variant
        >>> prompt = get_prompt("hhem_claim_decomposition", answer="BERT has 12 layers")

        >>> # Budget tier (GPT-4o-mini) loads BASE_PROMPT variant
        >>> prompt = get_prompt("answer_quality_eval", question="...", answer="...")
    """
    spec = get_model_for_task(task_name)
    is_gpt5 = _is_gpt5_family(spec.name)

    # Import appropriate task module and get variant
    if task_name == "hhem_claim_decomposition":
        from . import hhem_claim_decomposition as module
        template = module.GPT5_PROMPT if is_gpt5 else module.BASE_PROMPT

    elif task_name == "answer_quality_eval":
        from . import answer_quality_eval as module
        template = module.GPT5_PROMPT if is_gpt5 else module.BASE_PROMPT

    elif task_name == "retrieval_quality_eval":
        from . import retrieval_quality_eval as module
        template = module.GPT5_PROMPT if is_gpt5 else module.BASE_PROMPT

    elif task_name == "llm_reranking":
        from . import llm_reranking as module
        template = module.GPT5_PROMPT if is_gpt5 else module.BASE_PROMPT

    elif task_name == "answer_generation":
        from . import answer_generation as module
        template = module.GPT5_PROMPT if is_gpt5 else module.BASE_PROMPT

    elif task_name == "multi_agent_merge_reranking":
        from . import multi_agent_merge_reranking as module
        template = module.GPT5_PROMPT if is_gpt5 else module.BASE_PROMPT

    else:
        # Task not optimized - return empty string
        # Caller should use hardcoded prompt as fallback
        return ""

    # Format with kwargs if provided
    return template.format(**kwargs) if kwargs else template


def _is_gpt5_family(model_name: str) -> bool:
    """
    Detect if model is GPT-5 family.

    GPT-5 models need different prompting:
    - No few-shot examples (can hurt performance)
    - No Chain-of-Thought scaffolding (adds latency)
    - Concise, unambiguous instructions
    - Trust internal reasoning capabilities

    Args:
        model_name: Model identifier (e.g., "gpt-5.1", "gpt-4o-mini")

    Returns:
        True if GPT-5 family, False otherwise
    """
    model_lower = model_name.lower()
    return model_lower.startswith("gpt-5") or model_lower.startswith("gpt5")


__all__ = ["get_prompt"]
