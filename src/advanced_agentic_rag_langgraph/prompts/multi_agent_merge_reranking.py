"""
LLM Coverage-Aware Merge Reranking for Multi-Agent RAG.

Selects the optimal set of documents that together best cover
the original question's information needs.

Key difference from standard reranking:
- Considers the ORIGINAL question (not sub-queries)
- Knows what sub-queries were used (for coverage awareness)
- Selects a SET for completeness, not individual relevance scores

Research references:
- SetR (2024): Set-wise document selection for coverage
- PureCover (ICLR 2026): Coverage-based selection for multi-hop QA
- DynamicRAG (2025): Reasoning-based listwise reranking
"""

BASE_PROMPT = """You are selecting documents for a RAG system to answer a complex question.

ORIGINAL QUESTION: "{original_question}"

This question was decomposed into sub-queries for parallel retrieval:
{sub_queries_list}

CANDIDATE DOCUMENTS (from all retrieval workers):
{doc_list}

YOUR TASK:
Select exactly {k} documents that TOGETHER best cover ALL aspects of the original question.

SELECTION CRITERIA:
1. **Coverage**: Selected docs should address different sub-queries/aspects
2. **Relevance**: Each doc should contain SPECIFIC information that answers part of the question
   - Prioritize: Chunks with explicit answers, specific details, direct explanations
   - Deprioritize: Chunks that discuss the topic but don't contain the actual answer
   - Key distinction: "Related to the topic" != "Contains the answer"
3. **Non-redundancy**: Avoid selecting docs that cover the same information
4. **Complementarity**: Prefer docs that fill gaps left by others

IMPORTANT:
- You are selecting a SET, not ranking individuals
- A doc that covers an unaddressed aspect is more valuable than another doc on an already-covered aspect
- The goal is to maximize information coverage for answer generation

SELECTION STRATEGY:
1. Identify the key aspects/facets of the original question
2. For each aspect, identify which documents address it
3. Select documents to maximize coverage across ALL aspects
4. Prefer documents that uniquely cover an aspect over redundant ones

Return:
- selected_document_ids: List of exactly {k} document IDs (e.g., ["doc_0", "doc_3", "doc_7"])
- coverage_analysis: Brief explanation of how selected docs cover the question's aspects
- per_doc_reasoning: For each selected doc, which aspect/sub-query it primarily addresses"""


GPT5_PROMPT = """Select documents for answering a complex question.

ORIGINAL QUESTION: "{original_question}"

SUB-QUERIES (decomposed for parallel retrieval):
{sub_queries_list}

CANDIDATES:
{doc_list}

TASK: Select exactly {k} documents that TOGETHER best cover ALL aspects.

CRITERIA:
1. Coverage: Docs should address different sub-queries/aspects
2. Relevance: Each doc contains SPECIFIC information that answers part of the question
   - "Related to topic" != "Contains the answer"
3. Non-redundancy: Avoid duplicate information
4. Complementarity: Fill gaps left by other selections

KEY: Select a SET for coverage, not rank individuals.

Return:
- selected_document_ids: Exactly {k} document IDs (e.g., ["doc_0", "doc_3"])
- coverage_analysis: How selected docs cover the question's aspects
- per_doc_reasoning: Per doc, which aspect it addresses"""
