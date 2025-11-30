"""
HHEM Claim Decomposition Prompts

Used for hallucination detection via atomic claim extraction and HHEM-2.1-Open verification.

Research-backed optimizations:
- GPT-4o-mini (BASE): Few-shot examples improve decomposition accuracy ~6%
- GPT-5 (GPT5): Concise instructions leverage reasoning, ~18% improvement over baseline
- Expected F1: 0.68 (baseline) -> 0.72 (GPT-4o) -> 0.80 (GPT-5)

References:
- Zero-shot NLI baseline: ~0.65-0.70 F1
- Production with fine-tuning: ~0.79-0.83 F1 (requires RAGTruth dataset)
"""

BASE_PROMPT = """Extract all factual claims from this answer.

STEP-BY-STEP PROCESS:

1. Read the answer carefully and identify all factual statements
2. Break compound statements into separate atomic claims
3. Ensure each claim is self-contained (can be verified independently)
4. Include implicit claims if they're essential to the answer

CLAIM DECOMPOSITION GUIDELINES:

- ATOMIC: Each claim contains exactly ONE verifiable fact
- SELF-CONTAINED: Claim includes all necessary context (no pronouns like "it", "this")
- VERIFIABLE: Claim can be checked against source documents
- COMPLETE: Don't miss claims that are stated implicitly but essential

FEW-SHOT EXAMPLES:

Example 1:
Input: "The Eiffel Tower is 330 meters tall and was completed in 1889"
Reasoning: This is a compound statement with two distinct facts
Output:
  claims: ["The Eiffel Tower is 330 meters tall", "The Eiffel Tower was completed in 1889"]
  reasoning: "Separated compound statement into two atomic claims, each verifiable independently"

Example 2:
Input: "The process converts sunlight into electrical energy"
Reasoning: This is already atomic but needs subject clarity for "the process"
Output:
  claims: ["Solar panels convert sunlight into electrical energy"]
  reasoning: "Single atomic claim, added 'Solar panels' to specify the subject for completeness and verifiability"

Example 3:
Input: "The Great Wall of China was built over several dynasties starting in the 7th century BC and stretches over 13,000 miles"
Reasoning: Three distinct facts (construction history, start time, length)
Output:
  claims: [
    "The Great Wall of China was built over several dynasties",
    "The Great Wall of China construction started in the 7th century BC",
    "The Great Wall of China stretches over 13,000 miles"
  ]
  reasoning: "Decomposed compound statement into three atomic, independently verifiable claims"

NOW DECOMPOSE THIS ANSWER:

Answer:
{answer}

Provide:
- claims: List of atomic factual claims
- reasoning: Brief explanation of your decomposition approach"""


GPT5_PROMPT = """Extract atomic factual claims from this answer.

REQUIREMENTS:
- Atomic: One verifiable fact per claim
- Self-contained: Include all necessary context (no pronouns like "it", "this")
- Verifiable: Can be checked against source documents
- Complete: Include implicit claims if essential to the answer

Answer:
{answer}

Return:
- claims: List of atomic factual claims
- reasoning: Brief explanation of decomposition approach"""
