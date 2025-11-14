from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


class RankingResult(TypedDict):
    """Structured output schema for document reranking"""
    scores: list[float]  # 0-100 for each document
    reasoning: str  # Brief explanation of ranking decisions


class ReRanker:
    """
    Rerank documents using LLM-as-Judge pattern with metadata awareness.

    Uses structured output and considers document metadata (type, technical level,
    domain) to make context-aware relevance judgments.
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.structured_llm = self.llm.with_structured_output(RankingResult)

    def rank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """
        Rerank documents using metadata-aware relevance scoring.

        Args:
            query: User query
            documents: List of documents to rerank

        Returns:
            List of (document, score) tuples, sorted by relevance, limited to top_k
        """
        if not documents:
            return []

        # Build metadata-enriched document list
        doc_list = []
        for i, doc in enumerate(documents):
            meta = doc.metadata

            # Extract metadata context
            doc_context = f"{i+1}. "

            # Add type and level
            content_type = meta.get('content_type', 'unknown')
            tech_level = meta.get('technical_level', 'unknown')
            domain = meta.get('domain', 'general')

            doc_context += f"[Type: {content_type} | Level: {tech_level} | Domain: {domain}"

            # Add special features
            if meta.get('has_math'):
                doc_context += " | Has math"
            if meta.get('has_code'):
                doc_context += " | Has code"

            # Add source
            source = meta.get('source', 'unknown')
            doc_context += f" | Source: {source}]"

            # Add content preview
            doc_context += f"\n   Content: {doc.page_content[:200]}..."

            doc_list.append(doc_context)

        ranking_prompt = f"""Query: "{query}"

Documents with metadata:
{chr(10).join(doc_list)}

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
   - Simple queries (e.g., "What is X?") → beginner/intermediate docs preferred
   - Advanced queries (e.g., "Optimize X algorithm") → advanced docs preferred
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
- reasoning: 1-2 sentence explanation of your overall ranking approach
"""

        try:
            # Use structured output for reliable parsing
            result = self.structured_llm.invoke([HumanMessage(content=ranking_prompt)])
            scores = result["scores"]

            # Ensure we have a score for each document
            if len(scores) != len(documents):
                print(f"Warning: LLM returned {len(scores)} scores for {len(documents)} documents. Using fallback.")
                scores = [50] * len(documents)

        except Exception as e:
            # Fallback to neutral scores if LLM fails
            print(f"Warning: Reranking failed: {e}. Using neutral scores.")
            scores = [50] * len(documents)

        # Pair documents with scores and sort
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:self.top_k]
