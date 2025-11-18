"""
GPT-5 Model Parameter Smoke Test

Validates that GPT-5 model variants accept reasoning_effort and verbosity
parameters using the EXACT same pattern as the production pipeline.

Runtime: ~15-20 seconds
Cost: <$0.05 (uses minimal prompt)

Model Variants Tested (via tier configurations):
- Budget tier: gpt-4o-mini (baseline)
- Balanced tier: gpt-4o-mini + gpt-5-mini
- Premium tier: gpt-5.1 + gpt-5-mini + gpt-5-nano

Parameters Tested:
- reasoning_effort: minimal, low, medium, high (where configured)
- verbosity: low, medium, high (where configured)

This test validates API acceptance only, NOT output quality.
"""

import os
import sys
import warnings
import logging

# Suppress LangSmith warnings - MUST be before imports
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore", message=".*Failed to.*LangSmith.*")
warnings.filterwarnings("ignore", message=".*langsmith.*")
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.WARNING)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import production configuration (same as pipeline uses)
from advanced_agentic_rag_langgraph.core.model_config import (
    get_model_for_task,
    ModelTier,
    TIER_CONFIGS
)


# Minimal prompt to reduce cost
MINIMAL_PROMPT = "What is 2+2? Answer in one word."

# Representative tasks to test (covers different model/parameter combinations)
TASKS_TO_TEST = [
    "answer_generation",           # User-facing, critical
    "retrieval_quality_eval",      # Quality gate, medium reasoning
    "nli_claim_decomposition",     # High reasoning in premium
    "expansion_decision",          # Uses gpt-5-nano in premium
]


def test_task_with_tier(tier: ModelTier, task_name: str) -> dict:
    """
    Test a specific task in a specific tier using production pattern.

    Mirrors the exact instantiation pattern from production code:
    1. Get ModelSpec via get_model_for_task()
    2. Build model_kwargs with reasoning_effort (if present)
    3. Create ChatOpenAI instance
    4. Invoke with minimal prompt

    Args:
        tier: ModelTier enum value
        task_name: Task identifier

    Returns:
        Dict with test results: {
            "success": bool,
            "model": str,
            "reasoning_effort": str or None,
            "verbosity": str or None,
            "error": str or None,
            "response": str or None
        }
    """
    result = {
        "success": False,
        "model": None,
        "reasoning_effort": None,
        "verbosity": None,
        "error": None,
        "response": None
    }

    # Set tier environment variable (same as test_tier_comparison.py)
    os.environ["MODEL_TIER"] = tier.value

    try:
        # Get model spec using production function
        spec = get_model_for_task(task_name)
        result["model"] = spec.name
        result["reasoning_effort"] = spec.reasoning_effort
        result["verbosity"] = spec.verbosity

        # Create LLM instance (UPDATED PRODUCTION PATTERN - direct parameters)
        llm = ChatOpenAI(
            model=spec.name,
            temperature=spec.temperature,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity
        )

        # Invoke with minimal prompt
        response = llm.invoke([HumanMessage(content=MINIMAL_PROMPT)])
        result["response"] = response.content.strip()[:30]
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)[:150]

    return result


def print_test_result(tier_name: str, task_name: str, result: dict):
    """Print formatted test result."""
    status = "[OK]" if result["success"] else "[FAIL]"

    # Build parameter info
    params = []
    if result["reasoning_effort"]:
        params.append(f"reasoning={result['reasoning_effort']}")
    if result["verbosity"]:
        params.append(f"verbosity={result['verbosity']}")
    param_str = ", ".join(params) if params else "no params"

    print(f"  {status} {task_name:25} | {result['model']:15} | {param_str:30}", end="")

    if result["success"]:
        print(f" | Response: {result['response']}")
    else:
        print(f" | ERROR: {result['error']}")


def run_smoke_tests():
    """
    Main smoke test runner.

    Tests each tier's configuration using production get_model_for_task() pattern.
    """
    print("\n" + "="*100)
    print("GPT-5 PARAMETER SMOKE TEST")
    print("="*100)
    print(f"Testing {len(TIER_CONFIGS)} tiers with {len(TASKS_TO_TEST)} representative tasks each")
    print(f"Using PRODUCTION pattern: get_model_for_task() + ChatOpenAI instantiation")
    print()

    all_results = {}

    for tier in [ModelTier.BUDGET, ModelTier.BALANCED, ModelTier.PREMIUM]:
        print(f"\n{'='*100}")
        print(f"TIER: {tier.value.upper()}")
        print(f"{'='*100}")

        tier_results = {}

        for task_name in TASKS_TO_TEST:
            result = test_task_with_tier(tier, task_name)
            tier_results[task_name] = result
            print_test_result(tier.value, task_name, result)

        all_results[tier.value] = tier_results

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    for tier_name, tier_results in all_results.items():
        successes = sum(1 for r in tier_results.values() if r["success"])
        total = len(tier_results)
        status = "PASS" if successes == total else "PARTIAL" if successes > 0 else "FAIL"
        print(f"{tier_name:10} | {successes}/{total} tasks successful | {status}")

    # Check for verbosity implementation
    print("\n" + "="*100)
    print("VERBOSITY PARAMETER CHECK")
    print("="*100)

    verbosity_used = False
    for tier_name, tier_results in all_results.items():
        for task_name, result in tier_results.items():
            if result["verbosity"]:
                verbosity_used = True
                break

    if verbosity_used:
        print("[INFO] verbosity parameter is defined in model_config.py")
        print("[INFO] This test ATTEMPTED to pass it to ChatOpenAI")
        print("[INFO] Check results above to see if OpenAI API accepted it")
    else:
        print("[WARN] verbosity parameter NOT passed to ChatOpenAI")
        print("[WARN] Even though it's defined in ModelSpec, it's not being used")

    print("\n" + "="*100)
    print("[OK] Smoke test completed")
    print("="*100)

    return all_results


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or export it")
        sys.exit(1)

    results = run_smoke_tests()
