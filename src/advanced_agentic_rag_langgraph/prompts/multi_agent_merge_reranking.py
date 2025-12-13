"""LLM Document Scoring for Multi-Agent RAG Merge - scores multi-worker retrieval results."""

BASE_PROMPT = """Score documents for answering a question.

QUESTION: "{original_question}"

DOCUMENTS:
{doc_list}

Rate each document's relevance (0-100) for answering the question.

SCORING CRITERIA:

1. **Answer Content** (MOST IMPORTANT): Does the document contain SPECIFIC information that DIRECTLY answers the question?
   - PRIORITIZE: Definitions, methods, equations, experimental results, direct explanations
   - DEPRIORITIZE: General topic discussion, background context, tangential mentions
   - KEY: "Contains the answer" >> "Related to the topic"

2. **Specificity**: Concrete details over vague overviews
   - GOOD: Specific mechanisms, parameters, step-by-step processes, quantitative results
   - BAD: High-level summaries, "achieved good results" without specifics

3. **Completeness**: How much of the answer does this single document provide?

SCORING GUIDELINES:
- 90-100: Directly answers the question with specific details
- 75-89: Contains relevant answer content, good specificity
- 60-74: Related and useful, but lacks key details
- 40-59: Tangentially related, minimal answer content
- 20-39: Mostly off-topic or too general
- 0-19: Not relevant to the question

COMPLETENESS REQUIREMENT:
- You are scoring exactly {doc_count} documents
- Score each document IN ORDER from doc_0 to doc_{last_doc_idx}
- Your response MUST include exactly {doc_count} entries in scored_documents
- Expected document IDs: {expected_ids}

Return:
- scored_documents: List with one entry per document containing:
  - document_id: Document identifier string (e.g., "doc_0", "doc_1")
  - relevance_score: 0-100 score
  - reasoning: Brief explanation for this document's score
- overall_reasoning: Brief summary of ranking approach"""


GPT5_PROMPT = """Score documents for answering a question.

QUESTION: "{original_question}"

DOCUMENTS:
{doc_list}

Rate each document 0-100 for answering the question.

CRITERIA:
1. **Answer Content**: Specific info that directly answers (definitions, methods, results)
2. **Specificity**: Concrete details >> general discussion
3. **Completeness**: How much of the answer does this doc provide?

SCORING:
- 90-100: Direct answer with specifics
- 75-89: Good answer content
- 60-74: Related but lacks key details
- 40-59: Tangential
- 0-39: Off-topic

COMPLETENESS: Score exactly {doc_count} documents IN ORDER (doc_0 to doc_{last_doc_idx}).
Expected IDs: {expected_ids}

Return:
- scored_documents: Per doc: document_id, relevance_score (0-100), reasoning
- overall_reasoning: Brief summary"""


# ========== COVERAGE-BASED SELECTION (for complex/comparative queries) ==========

COVERAGE_PROMPT = """Select {k} documents that TOGETHER best answer the question.

QUESTION: "{original_question}"

ASPECTS (the question involves these facets):
{sub_queries_list}

DOCUMENTS:
{doc_list}

SELECTION GUIDANCE:
1. **Relevance first**: Select documents with specific, answer-relevant content
2. **Coverage-aware**: Consider whether the selected set addresses different facets of the question
3. **Avoid redundancy**: Prefer variety over multiple docs saying the same thing

Return:
- selected_document_ids: List of exactly {k} document IDs (e.g., ["doc_0", "doc_2", "doc_5"])
- selection_reasoning: Brief explanation of selection rationale"""


GPT5_COVERAGE_PROMPT = """Select {k} documents that TOGETHER best answer the question.

QUESTION: "{original_question}"

ASPECTS: {sub_queries_list}

DOCUMENTS:
{doc_list}

GUIDANCE:
1. Relevance first (specific, answer-relevant content)
2. Coverage-aware (consider different facets)
3. Avoid redundancy (prefer variety)

Return:
- selected_document_ids: Exactly {k} doc IDs
- selection_reasoning: Brief rationale"""
