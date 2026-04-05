"""Retrieval Quality Evaluation Prompts - evaluates context sufficiency (not just relevance)."""

BASE_PROMPT = """Query: {query}

Retrieved documents (top-k after reranking):
{docs_text}

Evaluate CONTEXT SUFFICIENCY: Do these documents contain the specific information needed to answer correctly?

CRITICAL DISTINCTION - Relevance vs Sufficiency:
- RELEVANT: Document discusses the same topic as the query
- SUFFICIENT: Document contains the SPECIFIC facts/details needed to answer

A document about "transistor physics" is RELEVANT to "Who invented the transistor?"
but NOT SUFFICIENT because it lacks the inventors' names.

SUFFICIENCY TEST (Answer these questions):
1. Can an expert definitively answer this query using ONLY the retrieved documents?
2. Are the specific entities, numbers, dates, or facts requested explicitly stated?
3. Would you need to "fill in gaps" or make assumptions to answer? If yes, NOT sufficient.

SCORING GUIDELINES (0-100, threshold 60):

- 80-100: SUFFICIENT - Documents contain explicit answers
  * All query-specific information explicitly stated (not implied)
  * Could directly quote passages to answer each part of query
  * No inference or assumption required

- 60-79: MOSTLY SUFFICIENT - Minor gaps that don't affect correctness
  * Core answer explicitly present in documents
  * Minor supporting details may be missing
  * Answer would be correct, just less detailed

- 40-59: INSUFFICIENT - Related but missing key information [WILL RETRY]
  * Documents are topically relevant (same domain/paper)
  * BUT: Specific requested information is NOT explicitly stated
  * Would need to infer, assume, or synthesize to answer
  * High hallucination risk if we proceed

- 0-39: IRRELEVANT - Wrong topic entirely [STRATEGY CHANGE]
  * Documents don't address query topic
  * Cannot construct any reasonable answer

FEW-SHOT EXAMPLES:

Example 1 (Sufficient - Score 85):
Query: "What year was the company founded and who were the founders?"
Documents: [Company history page stating "Founded in 2015 by Jane Smith and Robert Chen in Seattle"]
Evaluation:
  quality_score: 85
  reasoning: "Document explicitly states founding year (2015) and founders (Jane Smith, Robert Chen).
  Can directly quote the answer. No inference needed."
  issues: []

Example 2 (Insufficient - Score 50):
Query: "What year was the company founded and who were the founders?"
Documents: [Company overview discussing products, mission statement, current leadership team,
office locations, and recent acquisitions - but NO founding date or original founders mentioned]
Evaluation:
  quality_score: 50
  reasoning: "Documents discuss the company extensively but do NOT state when it was founded
  or who the founders were. Topically relevant (correct company) but missing the specific
  facts requested. Would need to guess or infer."
  issues: ["missing_key_info"]

Example 3 (Insufficient - Score 55):
Query: "Compare the pricing plans: what does the Pro tier cost vs the Enterprise tier?"
Documents: [Product documentation with Pro tier features and pricing ($49/month), FAQ about
billing cycles, upgrade process - but NO information about Enterprise tier pricing]
Evaluation:
  quality_score: 55
  reasoning: "Documents cover Pro tier pricing completely but Enterprise tier cost is
  not mentioned anywhere. Can only answer half the comparison. Related content present
  but cannot complete the requested comparison."
  issues: ["partial_coverage", "missing_key_info"]

Example 4 (Irrelevant - Score 25):
Query: "What database does the application use for storing user data?"
Documents: [Frontend React component documentation, CSS styling guide, UI/UX design principles]
Evaluation:
  quality_score: 25
  reasoning: "Documents discuss frontend/UI aspects but contain no backend or database
  information. Wrong part of the system entirely. Cannot answer database question."
  issues: ["wrong_domain", "off_topic"]

COMMON MISTAKES TO AVOID:
- DON'T score 60+ just because documents are "from the right paper/topic"
- DON'T score 60+ if you'd need to infer or synthesize the answer
- DON'T score 60+ for comparison queries if only one side is covered
- DO score below 60 if the specific fact/number/name requested is missing

STRUCTURED OUTPUT:
- quality_score (0-100): Based on SUFFICIENCY, not just relevance
- reasoning: 2-3 sentences on what IS vs ISN'T explicitly present
- issues: List problems (empty if none):
  * "missing_key_info": Specific requested information absent
  * "partial_coverage": Some query aspects covered, others missing
  * "incomplete_context": Related info but can't definitively answer
  * "wrong_domain": Documents from different topic area
  * "off_topic": Documents don't address query
- keywords_to_inject: If quality_score < 60, list of 2-5 specific keywords to inject into query.
  Empty list [] if quality_score >= 60.

KEYWORDS TO INJECT GUIDELINES (only if quality_score < 60):

CRITICAL ANTI-TAUTOLOGY CHECK:
Before suggesting keywords to inject:
1. List the KEY TERMS already present in the query
2. Do NOT suggest adding terms that are semantically equivalent to existing terms
3. Suggest DIFFERENT terms (synonyms, technical jargon, exact notation, domain-specific vocabulary)

EXAMPLES:
BAD: Query "company founding date" -> keywords_to_inject: ["when founded", "date established"]
GOOD: Query "company founding date" -> keywords_to_inject: ["incorporated", "2015", "startup year"]

BAD: Query "product pricing comparison" -> keywords_to_inject: ["price", "cost comparison"]
GOOD: Query "product pricing comparison" -> keywords_to_inject: ["$49/month", "Pro tier", "Enterprise plan", "annual discount"]

BAD: Query "author of the report" -> keywords_to_inject: ["who wrote", "writer"]
GOOD: Query "author of the report" -> keywords_to_inject: ["John Smith", "research team", "contributors", "published by"]

Return your evaluation as structured data."""


GPT5_PROMPT = """Query: {query}

Retrieved documents (top-k after reranking):
{docs_text}

Evaluate CONTEXT SUFFICIENCY (0-100, threshold 60).

KEY DISTINCTION:
- RELEVANT: Documents discuss same topic as query
- SUFFICIENT: Documents contain SPECIFIC facts needed to answer

SUFFICIENCY TEST:
1. Can expert definitively answer using ONLY these documents?
2. Are specific entities/numbers/facts explicitly stated (not implied)?
3. Would answering require inference or assumptions? If yes -> insufficient

SCORE:
- 80-100: SUFFICIENT - Explicit answers present, can quote directly
- 60-79: MOSTLY SUFFICIENT - Core answer explicit, minor gaps OK
- 40-59: INSUFFICIENT - Topically relevant but key info missing [RETRY]
- 0-39: IRRELEVANT - Wrong topic [STRATEGY CHANGE]

MISTAKES TO AVOID:
- Don't score 60+ for "right topic but wrong details"
- Don't score 60+ if you'd need to infer the answer
- Do score <60 if specific requested fact is absent

Issues: missing_key_info, partial_coverage, incomplete_context, wrong_domain, off_topic

ANTI-TAUTOLOGY CHECK (for keywords_to_inject):
- Do NOT suggest terms semantically equivalent to query terms
- Suggest DIFFERENT terms: synonyms, jargon, exact notation, domain vocabulary

Return:
- quality_score (0-100): Based on SUFFICIENCY not relevance
- reasoning (2-3 sentences: what IS vs ISN'T explicitly present)
- issues (list, empty if none)
- keywords_to_inject: If score < 60, list of 2-5 specific non-tautological keywords. Empty list if >= 60."""
