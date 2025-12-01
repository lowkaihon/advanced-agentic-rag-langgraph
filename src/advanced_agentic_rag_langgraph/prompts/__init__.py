"""Prompt template system with model-specific variant support (GPT-4o vs GPT-5)."""

from typing import Optional
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


def get_prompt(task_name: str, **kwargs) -> str:
    """Load prompt template with model-specific variant (BASE or GPT5) based on current tier."""
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
    """Detect if model is GPT-5 family (needs concise prompts, no few-shot)."""
    model_lower = model_name.lower()
    return model_lower.startswith("gpt-5") or model_lower.startswith("gpt5")


__all__ = ["get_prompt"]
