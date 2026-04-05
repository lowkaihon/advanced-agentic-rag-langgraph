"""LLM Metadata-Aware Reranking Prompts - second-stage LLM-as-judge scoring."""

BASE_PROMPT = """Query: "{query}"

Documents with metadata:
{doc_list}

Rate each document's relevance (0-100).

## 1. CONTENT RELEVANCE (Primary Criterion)

Does the document contain SPECIFIC information that answers the query?

PRIORITIZE: Explicit answers, specific details, direct explanations of mechanisms
DEPRIORITIZE: Mere topic mentions, tangential content, metadata/boilerplate

QUERY-TYPE SCORING:

| Query Type | Examples | Scoring Approach |
|------------|----------|------------------|
| FACTUAL | "What is X?", "Define Y" | Precision-first: exact answers 85+, background 40-60 |
| CONCEPTUAL | "How does X work?" | Coverage-first: mechanism explanations ARE answers (75+) |
| COMPARISON | "X vs Y", "difference between" | BOTH sides required: X-only 70-80, Y-only 70-80, comparison 85+ |
| ADAPTATION | "How does X adapt Y?" | Baseline required: original(Y) 70-80, adapted(X) 75-85, mechanism 85+ |

CRITICAL: For comparison/adaptation queries, foundational content about EITHER side scores 70+, not 40-60. You cannot understand a difference without understanding both sides.

## 2. DOCUMENT-QUERY FIT

| Factor | High Score | Low Score |
|--------|------------|-----------|
| Type match | Academic for research, tutorial for how-to | Mismatch (legal doc for technical query) |
| Level match | Complexity matches query sophistication | Advanced paper for basic question |
| Domain match | Exact topic alignment | Different domain entirely |

## 3. SCORING ANCHORS

90-100: Directly explains the mechanism/concept with specific details
70-85: Provides relevant perspective or partial explanation
50-65: Related but addresses different aspect or lacks depth
20-40: Mentions topic without meaningful contribution
0-15: Different topic entirely

FEW-SHOT EXAMPLE (Comparison Query):

Query: "How does distributed caching differ from local caching?"

Doc A: "Distributed caching systems like Redis store data across nodes for horizontal scaling."
-> Score: 88 (explains one side with specifics)

Doc B: "Local caching stores data in application memory using LRU caches."
-> Score: 78 (explains OTHER side - required context, NOT background)

Doc C: "Caching stores copies of data for faster access."
-> Score: 45 (generic definition, doesn't explain either side)

## 4. OUTPUT REQUIREMENTS

- Score exactly {doc_count} documents IN ORDER (doc_0 to doc_{last_doc_idx})
- Expected IDs: {expected_ids}

Return:
- scored_documents: List of objects, each with document_id, relevance_score, reasoning (1-2 sentences)
- overall_reasoning: Brief ranking approach summary"""


GPT5_PROMPT = """Query: "{query}"

Documents with metadata:
{doc_list}

Rate each document's relevance (0-100).

CRITERIA:

1. Content relevance: Does content contain SPECIFIC information that answers the query?
   - Prioritize: Explicit answers, specific details, direct explanations
   - Deprioritize: Topic mentions without substance

   QUERY-TYPE SCORING:
   - FACTUAL ("What is X?"): Precision-first, exact answers 85+
   - CONCEPTUAL ("How does X work?"): Mechanism explanations ARE answers (75+)
   - COMPARISON ("X vs Y"): Both sides required - X-only 70+, Y-only 70+, comparison 85+
   - ADAPTATION ("How X adapts Y"): Baseline required - original 70+, adapted 75+, mechanism 85+

2. Document-query fit: Type, level, and domain alignment

SCORING:
- 90-100: Perfect (answers directly with specifics)
- 70-85: Excellent (relevant perspective or partial explanation)
- 50-65: Good (related, different aspect)
- 20-40: Low (mentions topic only)
- 0-15: Not relevant

COMPLETENESS: Score exactly {doc_count} documents IN ORDER (doc_0 to doc_{last_doc_idx}).
Expected IDs: {expected_ids}

Return:
- scored_documents: List with document_id, relevance_score, reasoning
- overall_reasoning: Brief ranking approach summary"""
