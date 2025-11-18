"""
Retrieval Quality Evaluation Prompts

Evaluates retrieved documents to determine if sufficient for answer generation.

Research-backed optimizations:
- GPT-4o-mini (BASE): Explicit scoring rubric + examples improve alignment ~10%
- GPT-5 (GPT5): Concise criteria + direct assessment leverage reasoning, ~12% improvement
- Expected correlation with human judges: 0.82 (baseline) -> 0.88-0.92 (optimized)

Evaluation dimensions:
- Coverage: How many query aspects are addressed?
- Completeness: Can query be fully answered?
- Relevance: Are documents on-topic?

Scoring thresholds:
- 80-100: Excellent (proceed immediately)
- 60-79: Good (acceptable, will proceed)
- 40-59: Fair (retry with query rewriting)
- 0-39: Poor (needs strategy change)
"""

BASE_PROMPT = """Query: {query}

Retrieved documents (top 5 after reranking):
{docs_text}

Evaluate retrieval quality to determine if these documents are sufficient for answer generation.

EVALUATION CRITERIA:

1. Coverage: How many aspects of the query are addressed?
   - Multi-aspect query (e.g., "advantages AND disadvantages"): Both aspects needed
   - Single-aspect query: Core information must be present
   - Consider: Are all parts of the question answered by the documents?

2. Completeness: Can the query be fully answered with these documents?
   - Complete information present: Documents contain everything needed
   - Partial information: Some details present but gaps exist
   - Insufficient: Cannot answer without additional sources

3. Relevance: Are documents on-topic and directly useful?
   - High relevance: Documents directly address query topic
   - Mixed relevance: Some docs relevant, others tangential
   - Low relevance: Documents off-topic or only peripherally related

SCORING GUIDELINES (0-100 scale, aligned with routing threshold of 60):

- 80-100: EXCELLENT - Proceed to answer generation immediately
  * All/most query aspects directly addressed
  * Complete information for full answer
  * All documents highly relevant to query

- 60-79: GOOD - Acceptable for answer generation [THRESHOLD: Will proceed]
  * Key query aspects covered (may have minor gaps)
  * Sufficient information for complete answer
  * Most documents relevant, minimal noise

- 40-59: FAIR - Requires query rewriting [THRESHOLD: Will retry if attempts < 2]
  * Partial coverage, key information missing
  * Incomplete information, gaps in answer
  * Documents tangential or only partially relevant

- 0-39: POOR - Inadequate retrieval, needs strategy change
  * Wrong domain or off-topic documents
  * Cannot answer query with current results
  * Most/all documents irrelevant

STRUCTURED OUTPUT:

- quality_score (0-100): Aggregate score following guidelines above

- reasoning: 2-3 sentences explaining:
  * Which aspects are covered vs missing
  * Whether information is complete for answering
  * Relevance quality of documents

- issues: List specific problems (empty list if none):
  * "missing_key_info": Required information not in documents (specify what is missing)
  * "partial_coverage": Some query aspects covered, others missing (list missing aspects)
  * "incomplete_context": Context lacks necessary details to fully answer query
  * "wrong_domain": Documents from unrelated topic area
  * "insufficient_depth": Surface-level info only, lacks detail
  * "off_topic": Documents irrelevant to query
  * "mixed_relevance": Combination of relevant and irrelevant docs

IMPORTANT: If key information or query aspects are missing, explicitly include "partial_coverage"
or "missing_key_info" in the issues list. This assessment is critical for routing decisions.

Return your evaluation as structured data."""


GPT5_PROMPT = """Query: {query}

Retrieved documents (top 5 after reranking):
{docs_text}

Evaluate if these documents sufficiently answer the query.

CRITERIA:

Coverage: Are all query aspects addressed?
- Multi-aspect queries need all parts covered
- Single-aspect queries need core information

Completeness: Can query be fully answered?
- Complete: All information present
- Partial: Some gaps exist
- Insufficient: Missing key details

Relevance: Are documents on-topic?
- High: Directly address query
- Mixed: Some relevant, some tangential
- Low: Off-topic or peripheral

SCORING (0-100, threshold 60):
- 80-100: Excellent (proceed immediately)
- 60-79: Good (acceptable, will proceed)
- 40-59: Fair (retry with rewriting)
- 0-39: Poor (needs strategy change)

Issues (select applicable):
- missing_key_info, partial_coverage, incomplete_context
- wrong_domain, insufficient_depth, off_topic, mixed_relevance

Return:
- quality_score (0-100)
- reasoning (2-3 sentences: coverage, completeness, relevance)
- issues (list, empty if none; include partial_coverage or missing_key_info if key information missing)"""
