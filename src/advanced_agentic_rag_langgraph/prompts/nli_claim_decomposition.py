"""
NLI Claim Decomposition Prompts

Used for hallucination detection via atomic claim extraction and NLI verification.

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
Input: "BERT has 12 layers and uses transformers"
Reasoning: This is a compound statement with two distinct facts
Output:
  claims: ["BERT has 12 layers", "BERT uses transformers"]
  reasoning: "Separated compound statement into two atomic claims, each verifiable independently"

Example 2:
Input: "The attention mechanism allows models to focus on relevant parts"
Reasoning: This is already atomic but needs context for "relevant parts"
Output:
  claims: ["The attention mechanism allows models to focus on relevant parts of the input"]
  reasoning: "Single atomic claim, added 'of the input' for completeness and verifiability"

Example 3:
Input: "GPT-3 was released by OpenAI in 2020 and has 175 billion parameters"
Reasoning: Three distinct facts (who, when, size)
Output:
  claims: [
    "GPT-3 was released by OpenAI",
    "GPT-3 was released in 2020",
    "GPT-3 has 175 billion parameters"
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
