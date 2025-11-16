"""
LLM-based document profiling using structured output.

This module uses LLMs to profile full documents before chunking,
generating rich metadata that informs retrieval strategy selection.
"""

import re
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class DocumentProfile(TypedDict):
    """Structured output schema for document profiling"""
    doc_type: Literal[
        # Academic (6 types)
        "research_paper", "thesis", "dissertation",
        "conference_paper", "journal_article", "literature_review",
        # Educational (5 types)
        "tutorial", "course_material", "textbook",
        "lecture_notes", "study_guide",
        # Technical (4 types)
        "api_reference", "technical_specification",
        "architecture_document", "system_design",
        # Business (4 types)
        "whitepaper", "case_study", "business_report", "proposal",
        # Legal (3 types)
        "legal_document", "contract", "policy_document",
        # General (6 types)
        "blog_post", "article", "guide", "manual", "faq", "documentation",
        # Fallback
        "other"
    ]
    doc_type_description: str  # Clarification or custom description if type="other"
    technical_density: float  # 0.0-1.0
    reading_level: Literal["beginner", "intermediate", "advanced"]
    domain_tags: list[str]  # Top 3 domains
    best_retrieval_strategy: Literal["semantic", "keyword", "hybrid"]
    strategy_confidence: float  # 0.0-1.0
    has_math: bool
    has_code: bool
    summary: str  # 2-3 sentence summary
    key_concepts: list[str]  # Top 5 concepts


class DocumentProfiler:
    """
    Profile documents using LLM with structured output.

    Analyzes full documents before chunking to extract:
    - Document type and structure
    - Technical density and reading level
    - Domain tags and key concepts
    - Optimal retrieval strategy with confidence
    - Presence of math and code
    - Document summary
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize LLM-based document profiler.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
            temperature: Sampling temperature (0 for deterministic)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(DocumentProfile)

    def _detect_signals(self, doc_text: str) -> dict:
        """
        Quick regex-based signal detection (zero-cost preprocessing).

        Detects presence of code and math patterns before LLM profiling,
        allowing LLM to confirm and classify rather than discover from scratch.

        Args:
            doc_text: Full document text

        Returns:
            Dictionary with has_code_signal and has_math_signal booleans
        """
        return {
            'has_code_signal': bool(re.search(
                r'```|^def |^class |^function|import |#include|public class|private |protected ',
                doc_text,
                re.MULTILINE
            )),
            'has_math_signal': bool(re.search(
                r'\$.*?\$|\\[.*?\\]|\\begin\{equation\}|∑|∏|∫|≤|≥|∂|∇',
                doc_text
            )),
        }

    def _stratified_sample(self, doc_text: str, target_tokens: int = 5000) -> str:
        """
        Sample document using stratified positional strategy.

        Research-backed approach:
        - First 30% of document: 40-50% of token budget (intro, abstract, metadata)
        - Last 20% of document: 20-25% of token budget (conclusions, key takeaways)
        - Middle sections: 25-30% of token budget (body content, technical details)

        This maximizes coverage of metadata that appears at document boundaries
        while ensuring middle sections are sampled for code/math/concepts.

        Args:
            doc_text: Full document text
            target_tokens: Target token count (default: 5000 for optimal cost/accuracy)

        Returns:
            Stratified sample of document with section markers
        """
        # Rough estimate: 1 token ≈ 4 chars
        target_chars = target_tokens * 4  # ~20,000 chars for 5,000 tokens

        doc_length = len(doc_text)

        # If document is shorter than target, use it all
        if doc_length <= target_chars:
            return doc_text

        # Calculate section boundaries (positional strategy)
        first_30_pct = int(doc_length * 0.30)
        last_20_pct_start = int(doc_length * 0.80)

        # Token budget allocation (in characters)
        first_budget = int(target_chars * 0.45)  # 45% for first 30%
        last_budget = int(target_chars * 0.22)   # 22% for last 20%
        middle_budget = target_chars - first_budget - last_budget  # ~33% for middle

        # Sample first section (intro, abstract, early content)
        first_section = doc_text[:min(first_30_pct, first_budget)]

        # Sample last section (conclusions, appendices, final content)
        last_section = doc_text[max(last_20_pct_start, doc_length - last_budget):]

        # Sample middle section (body, methods, results)
        middle_start = first_30_pct
        middle_end = last_20_pct_start
        middle_available = middle_end - middle_start

        if middle_available > middle_budget:
            # Sample from middle of the middle section
            middle_sample_start = middle_start + (middle_available - middle_budget) // 2
            middle_section = doc_text[middle_sample_start:middle_sample_start + middle_budget]
        else:
            middle_section = doc_text[middle_start:middle_end]

        # Combine sections with clear separators
        return f"{first_section}\n\n[... middle section sampled ...]\n\n{middle_section}\n\n[... final section sampled ...]\n\n{last_section}"

    def profile_document(self, doc_text: str, doc_id: str = None) -> DocumentProfile:
        """
        Profile a full document using LLM analysis with stratified sampling.

        Uses research-backed stratified positional sampling (5000 tokens) and
        regex-based signal pre-detection for optimal accuracy/cost trade-off.

        Args:
            doc_text: Full document text (before chunking)
            doc_id: Optional document identifier

        Returns:
            DocumentProfile with structured metadata
        """
        # Quick signal detection (zero-cost preprocessing)
        signals = self._detect_signals(doc_text)

        # Stratified sampling for comprehensive coverage
        content = self._stratified_sample(doc_text, target_tokens=5000)
        sampled = len(doc_text) > 20000  # ~5000 tokens

        prompt = f"""Analyze this document comprehensively for retrieval optimization.

Document ID: {doc_id or 'unknown'}
Content (stratified sample from full document): {content}
{'... [sampled from full document, original length: ' + str(len(doc_text)) + ' chars]' if sampled else ''}

Pre-detected signals (confirm and classify if present):
- Code patterns detected: {signals['has_code_signal']}
- Math notation detected: {signals['has_math_signal']}

Profile the document according to these criteria:

1. **doc_type**: What kind of document is this? Choose the MOST SPECIFIC type:

   ACADEMIC (for scholarly research and academic work):
   - research_paper: Published academic research with methodology, results, citations (general research)
   - conference_paper: Paper presented at academic conference (e.g., NeurIPS, ACL, CVPR)
   - journal_article: Peer-reviewed article published in academic journal
   - thesis: Master's thesis or comprehensive academic work
   - dissertation: Doctoral dissertation or PhD research
   - literature_review: Survey or systematic review of existing research

   EDUCATIONAL (for learning and teaching materials):
   - tutorial: Step-by-step instructional guide with hands-on examples
   - course_material: Lecture slides, course notes, homework, syllabi
   - textbook: Educational textbook chapter or reference material
   - lecture_notes: Notes from lectures, talks, or presentations
   - study_guide: Exam preparation materials, summary sheets, quick references

   TECHNICAL (for system and software documentation):
   - api_reference: API, function, class, or method documentation
   - technical_specification: Formal technical specs (RFC, protocol documentation, standards)
   - architecture_document: System architecture, design docs, architectural patterns
   - system_design: Software design patterns, implementation guides, technical proposals

   BUSINESS (for corporate and business documents):
   - whitepaper: Industry analysis, thought leadership, business strategy
   - case_study: Real-world example, success story, implementation report
   - business_report: Financial reports, market analysis, business metrics
   - proposal: Business proposals, RFPs, project proposals

   LEGAL (for legal and compliance documents):
   - legal_document: Legal briefs, terms of service, privacy policies
   - contract: Formal agreements, legal contracts, SLAs
   - policy_document: Company policies, procedures, compliance guidelines

   GENERAL (for general content):
   - blog_post: Informal blog article, opinion piece
   - article: News article, magazine piece, web content
   - guide: General how-to guide, user manual, handbook
   - manual: Reference manual, user guide, instruction manual
   - faq: Frequently asked questions document
   - documentation: General technical or product documentation

   - other: If none of the above fit (you MUST provide description in doc_type_description)

2. **doc_type_description**:
   - If you chose a specific type: Provide a 1-sentence clarification (e.g., "Conference paper on transformer architectures presented at NeurIPS 2017")
   - If you chose "other": Describe the document type in detail (e.g., "Product specification sheet for medical device")

3. **technical_density**: How technical is the content? (0.0-1.0)
   - 0.0-0.2: General audience, no jargon (e.g., news article, blog post)
   - 0.3-0.5: Intermediate, some technical terms (e.g., tutorial, business report)
   - 0.6-0.8: Technical, domain-specific language (e.g., technical docs, academic papers)
   - 0.9-1.0: Expert-only, heavy mathematical/technical content (e.g., research papers, specifications)

4. **reading_level**: Target audience expertise?
   - beginner: Introductory, assumes little to no background knowledge
   - intermediate: Some domain knowledge expected, familiar with basics
   - advanced: Expert-level, assumes deep background and domain expertise

5. **domain_tags**: What domains/topics does this cover? (list top 3, ordered by relevance)
   Examples by field:
   - AI/ML: machine_learning, deep_learning, nlp, computer_vision, transformers, reinforcement_learning
   - Computer Science: algorithms, data_structures, databases, networking, security, distributed_systems
   - Math/Stats: statistics, linear_algebra, calculus, optimization, probability
   - Business: finance, marketing, strategy, operations, management
   - Science: physics, chemistry, biology, medicine, neuroscience
   - Engineering: electrical_engineering, mechanical_engineering, civil_engineering
   - General: any specific topic or field

6. **best_retrieval_strategy**: What search approach works best for this content?
   - semantic: For understanding concepts, finding similar ideas, explanation-heavy content
   - keyword: For exact lookups, technical terms, API names, specific citations, proper nouns
   - hybrid: Mixed content, comparisons, general use (when unsure, choose this)

7. **strategy_confidence**: How confident are you in the strategy choice? (0.0-1.0)
   - 0.9-1.0: Very confident, clear document type and optimal strategy
   - 0.7-0.8: Moderately confident, good match but some ambiguity
   - 0.5-0.6: Uncertain, mixed signals or unclear document type
   - Below 0.5: Very uncertain (rare, use hybrid in these cases)

8. **has_math**: Does the document contain mathematical equations, formulas, or formal notation?
   - True: Contains LaTeX equations, mathematical symbols, formal mathematical content
   - False: No mathematical notation

9. **has_code**: Does the document contain code examples, programming syntax, or code snippets?
   - True: Contains code blocks, function definitions, programming examples
   - False: No code content

10. **summary**: Write a 2-3 sentence summary of what this document is about.
    Include: main topic, purpose, and key contribution/takeaway

11. **key_concepts**: List the top 5 key concepts, terms, or topics covered.
    Be specific (e.g., "multi-head attention" not just "attention")

IMPORTANT INSTRUCTIONS:
- Be SPECIFIC with doc_type (choose conference_paper over research_paper if it's from a conference)
- Always provide doc_type_description with meaningful clarification
- Consider the ENTIRE document context, not just the beginning
- Domain tags should be specific and relevant (avoid generic tags like "general" unless truly general)
- If document is truly novel or unique, use "other" and describe it well
"""

        try:
            profile = self.structured_llm.invoke([HumanMessage(content=prompt)])
            return profile
        except Exception as e:
            # Fallback to safe defaults if LLM call fails
            print(f"Warning: LLM profiling failed for {doc_id}: {e}")
            return self._get_fallback_profile(doc_text, doc_id)

    def _get_fallback_profile(self, doc_text: str, doc_id: str = None) -> DocumentProfile:
        """
        Generate fallback profile if LLM call fails.

        Uses simple heuristics as safety net.
        """
        return {
            "doc_type": "other",
            "doc_type_description": f"Unknown document type (LLM profiling failed for {doc_id or 'unknown'})",
            "technical_density": 0.5,
            "reading_level": "intermediate",
            "domain_tags": ["general"],
            "best_retrieval_strategy": "hybrid",
            "strategy_confidence": 0.3,
            "has_math": "$" in doc_text or "equation" in doc_text.lower(),
            "has_code": "```" in doc_text or "def " in doc_text or "class " in doc_text,
            "summary": f"Document {doc_id or 'unknown'} (profiling failed, using fallback)",
            "key_concepts": ["general_content"]
        }

    def profile_corpus(self, documents: list[tuple[str, str]]) -> dict[str, DocumentProfile]:
        """
        Profile multiple documents.

        Args:
            documents: List of (doc_id, doc_text) tuples

        Returns:
            Dictionary mapping doc_id to DocumentProfile
        """
        profiles = {}
        for doc_id, doc_text in documents:
            profiles[doc_id] = self.profile_document(doc_text, doc_id)
        return profiles
