"""
LLM Metadata-Aware Reranking Prompts

Second-stage reranking using LLM-as-judge with metadata intelligence.

Research-backed optimizations:
- GPT-4o-mini (BASE): Explicit document type taxonomy + scoring examples improve ranking ~8%
- GPT-5 (GPT5): Streamlined criteria + concise type definitions leverage reasoning, ~10% improvement
- Expected reranking accuracy: 0.85 (baseline) -> 0.88 (GPT-4o) -> 0.93 (GPT-5)

Evaluation dimensions:
- Content relevance: Does content address the query?
- Document type appropriateness: Right KIND of document?
- Technical level match: Complexity matches query sophistication?
- Domain alignment: Document domain matches query topic?

Applied after first-stage cross-encoder reranking to top-15 candidates.
"""

BASE_PROMPT = """Query: "{query}"

Documents with metadata:
{doc_list}

Rate each document's relevance (0-100) considering:

1. **Content relevance**: Does the content directly answer or address the query?

2. **Document type appropriateness**: Is this the right KIND of document for this query?

   ACADEMIC (scholarly research):
   - research_paper, conference_paper, journal_article, thesis, dissertation, literature_review
   - Best for: Deep understanding, methodologies, theoretical concepts, research findings
   - Good for queries: "What is X?", "How does X work theoretically?", "Research on X"

   EDUCATIONAL (learning materials):
   - tutorial, course_material, textbook, lecture_notes, study_guide
   - Best for: Learning step-by-step, educational content, beginner/intermediate explanations
   - Good for queries: "How to learn X?", "Tutorial on X", "Explain X for beginners"

   TECHNICAL (implementation and specs):
   - api_reference, technical_specification, architecture_document, system_design
   - Best for: Implementation details, API usage, technical specifications, system architecture
   - Good for queries: "How to use X function?", "X API documentation", "Architecture of X"

   BUSINESS (corporate documents):
   - whitepaper, case_study, business_report, proposal
   - Best for: Industry insights, real-world examples, market analysis, business applications
   - Good for queries: "X use cases", "Business value of X", "X case study"

   LEGAL (legal/compliance):
   - legal_document, contract, policy_document
   - Best for: Legal requirements, compliance, policies, terms
   - Good for queries: "Legal aspects of X", "Compliance with X", "X policy"

   GENERAL (articles and guides):
   - blog_post, article, guide, manual, faq, documentation
   - Best for: General information, how-to guides, quick references, FAQs
   - Good for queries: General questions, practical guides, quick lookups

3. **Technical level match**: Does the document's complexity match the query's sophistication?
   - Simple queries (e.g., "What is X?"): beginner/intermediate docs preferred
   - Advanced queries (e.g., "Optimize X algorithm"): advanced docs preferred
   - Mismatch penalty: Don't give advanced papers for basic questions, or vice versa

4. **Domain alignment**: Does the document's domain match the query's topic?
   - Strong match: Document domain exactly matches query topic
   - Partial match: Related but not identical domain
   - No match: Different domain (penalize heavily)

SCORING GUIDELINES:
- 90-100: Perfect match (right type, right level, right domain, answers query directly)
- 75-89: Excellent match (right type and domain, answers query well)
- 60-74: Good match (relevant but not ideal type or level)
- 40-59: Moderate relevance (somewhat relevant but wrong type or level)
- 20-39: Low relevance (tangentially related, wrong document type)
- 0-19: Not relevant (wrong topic, wrong type, doesn't answer query)

IMPORTANT:
- Do NOT consider how documents were retrieved (semantic/keyword/hybrid)
- Judge ONLY the intrinsic quality and appropriateness for THIS specific query
- Prioritize documents whose TYPE matches the query intent (e.g., tutorial for "how to" questions)

Provide:
- scores: List of relevance scores (0-100) for each document, in order
- reasoning: 1-2 sentence explanation of your overall ranking approach"""


GPT5_PROMPT = """Query: "{query}"

Documents with metadata:
{doc_list}

Rate each document's relevance (0-100).

CRITERIA:

1. Content relevance: Does content address the query?

2. Document type match: Right KIND for this query?
   - Academic: research papers, journal articles (theoretical understanding, research)
   - Educational: tutorials, textbooks (learning, explanations)
   - Technical: API docs, specs (implementation, architecture)
   - Business: whitepapers, case studies (use cases, market analysis)
   - Legal: policies, contracts (compliance, terms)
   - General: articles, guides (general info, how-to)

3. Technical level match: Complexity matches query sophistication?
   - Simple queries prefer beginner/intermediate docs
   - Advanced queries need advanced content

4. Domain alignment: Document domain matches query topic?
   - Strong, partial, or no match

SCORING:
- 90-100: Perfect (right type, level, domain, answers directly)
- 75-89: Excellent (right type and domain, answers well)
- 60-74: Good (relevant, not ideal type/level)
- 40-59: Moderate (somewhat relevant, wrong type/level)
- 20-39: Low (tangential, wrong type)
- 0-19: Not relevant (wrong topic/type)

Return:
- scores: List of 0-100 scores for each document, in order
- reasoning: 1-2 sentences on ranking approach"""
