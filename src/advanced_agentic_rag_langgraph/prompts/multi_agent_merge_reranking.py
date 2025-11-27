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
Select exactly {k} documents that TOGETHER provide the best information to answer the original question.

SELECTION CRITERIA (in priority order):

1. **Answer Content** (MOST IMPORTANT): Prioritize docs containing SPECIFIC factual information that DIRECTLY answers parts of the question:
   - Look for: definitions, methods, equations, experimental results, direct comparisons
   - "Contains the answer" is far more valuable than "Related to the topic"
   - A chunk explaining HOW something works >> a chunk that merely mentions it

2. **Depth over Breadth**: Multiple chunks from the SAME source with complementary answer details are BETTER than diverse but superficial coverage. Complex questions often need consecutive/nearby chunks from one source.

3. **Evidence Quality**: Prefer chunks with concrete details over general topic discussions:
   - GOOD: Specific mechanisms, parameters, step-by-step processes, quantitative results
   - BAD: High-level overviews, vague references, "achieved good results" without specifics

4. **Coverage** (secondary): AFTER satisfying above criteria, ensure different aspects of the question are addressed. But NEVER sacrifice answer content for coverage diversity.

IMPORTANT:
- Sub-query coverage is a SECONDARY goal - prioritize answer content first
- Do NOT penalize multiple chunks from the same source if they contain different answer details
- The goal is to maximize information needed for answer generation, not topic diversity

Return:
- selected_document_ids: List of exactly {k} document IDs (e.g., ["doc_0", "doc_3", "doc_7"])
- coverage_analysis: Brief explanation of how selected docs provide answer content
- per_doc_reasoning: For each selected doc, what specific answer content it provides"""


GPT5_PROMPT = """Select documents for answering a complex question.

ORIGINAL QUESTION: "{original_question}"

SUB-QUERIES (decomposed for parallel retrieval):
{sub_queries_list}

CANDIDATES:
{doc_list}

TASK: Select exactly {k} documents with the best ANSWER CONTENT.

CRITERIA (priority order):
1. **Answer Content**: Docs with SPECIFIC factual info that directly answers parts of the question (definitions, methods, results, comparisons)
2. **Depth over Breadth**: Multiple chunks from same source with complementary details >> diverse superficial coverage
3. **Evidence Quality**: Concrete details >> general topic discussion
4. **Coverage**: Secondary - ensure aspects covered, but never sacrifice answer content for diversity

KEY: "Contains the answer" >> "Related to topic". Multiple chunks from same source is OK if they have different answer details.

Return:
- selected_document_ids: Exactly {k} document IDs (e.g., ["doc_0", "doc_3"])
- coverage_analysis: How selected docs provide answer content
- per_doc_reasoning: Per doc, what specific answer content it provides"""
