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
        "research_paper", "thesis", "dissertation",
        "conference_paper", "journal_article", "literature_review",
        "tutorial", "course_material", "textbook",
        "lecture_notes", "study_guide",
        "api_reference", "technical_specification",
        "architecture_document", "system_design",
        "whitepaper", "case_study", "business_report", "proposal",
        "legal_document", "contract", "policy_document",
        "blog_post", "article", "guide", "manual", "faq", "documentation",
        "other"
    ]
    doc_type_description: str
    technical_density: float
    reading_level: Literal["beginner", "intermediate", "advanced"]
    domain_tags: list[str]
    best_retrieval_strategy: Literal["semantic", "keyword", "hybrid"]
    strategy_confidence: float
    has_math: bool
    has_code: bool
    summary: str
    key_concepts: list[str]


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
        """Initialize LLM-based document profiler."""
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(DocumentProfile)

    def _detect_signals(self, doc_text: str) -> dict:
        """
        Quick regex-based signal detection (zero-cost preprocessing).

        Detects presence of code and math patterns before LLM profiling,
        allowing LLM to confirm and classify rather than discover from scratch.
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
        target_chars = target_tokens * 4
        doc_length = len(doc_text)

        if doc_length <= target_chars:
            return doc_text

        first_30_pct = int(doc_length * 0.30)
        last_20_pct_start = int(doc_length * 0.80)

        first_budget = int(target_chars * 0.45)
        last_budget = int(target_chars * 0.22)
        middle_budget = target_chars - first_budget - last_budget

        first_section = doc_text[:min(first_30_pct, first_budget)]
        last_section = doc_text[max(last_20_pct_start, doc_length - last_budget):]

        middle_start = first_30_pct
        middle_end = last_20_pct_start
        middle_available = middle_end - middle_start

        if middle_available > middle_budget:
            middle_sample_start = middle_start + (middle_available - middle_budget) // 2
            middle_section = doc_text[middle_sample_start:middle_sample_start + middle_budget]
        else:
            middle_section = doc_text[middle_start:middle_end]

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

<document_metadata>
Document ID: {doc_id or 'unknown'}
{'Sampled from full document (original length: ' + str(len(doc_text)) + ' chars)' if sampled else 'Full document provided'}
</document_metadata>

<document_sample>
{content}
</document_sample>

<pre_detected_signals>
Code patterns detected: {signals['has_code_signal']}
Math notation detected: {signals['has_math_signal']}
</pre_detected_signals>

<few_shot_examples>
Example 1 - Legal Document:
Document: "SOFTWARE LICENSE AGREEMENT. This Agreement is entered into as of January 1, 2024, by and between Licensor and Licensee. 1. GRANT OF LICENSE. Licensor hereby grants to Licensee a non-exclusive, non-transferable license..."
Profile:
- doc_type: legal_document
- doc_type_description: "Software license agreement establishing terms between licensor and licensee"
- technical_density: 0.3
- reading_level: intermediate
- domain_tags: ["legal", "software", "contracts"]
- best_retrieval_strategy: keyword
- strategy_confidence: 0.85
- has_math: False
- has_code: False
- summary: "Legal agreement defining software licensing terms, restrictions, and obligations between parties. Focuses on rights, limitations, and compliance requirements."
- key_concepts: ["license_grant", "intellectual_property", "terms_and_conditions", "compliance", "liability"]
Reasoning: Formal legal language with numbered clauses, contract structure, legal terminology like "hereby grants", "non-exclusive". Keyword search optimal for finding specific clauses and legal terms.

Example 2 - Educational Tutorial:
Document: "Beginner's Guide to Photography. Chapter 1: Understanding Aperture. Aperture is one of three elements that control exposure. Think of it like the pupil of your eye - it opens wider in dim light and closes in bright light. Let's start with the basics: What is f-stop? The f-stop number tells you how wide the aperture opening is..."
Profile:
- doc_type: tutorial
- doc_type_description: "Beginner photography tutorial teaching fundamental camera settings with practical examples"
- technical_density: 0.4
- reading_level: beginner
- domain_tags: ["photography", "education", "camera_techniques"]
- best_retrieval_strategy: semantic
- strategy_confidence: 0.9
- has_math: False
- has_code: False
- summary: "Step-by-step photography tutorial explaining aperture settings for beginners. Uses analogies and simple explanations to teach camera fundamentals without assuming prior knowledge."
- key_concepts: ["aperture", "f_stop", "exposure", "camera_settings", "beginner_techniques"]
Reasoning: Instructional tone, uses analogies ("like the pupil of your eye"), assumes no prior knowledge, step-by-step structure. Semantic search best for conceptual understanding questions.

Example 3 - Business Report:
Document: "Q3 2023 Financial Performance Report. Executive Summary: Revenue increased 23% YoY to $145M, driven by enterprise segment growth. Key Metrics: - ARR: $520M (+31% YoY) - Gross Margin: 72% (up 3pp) - Customer Acquisition Cost: $12K (down 15%) Enterprise segment represented 68% of new bookings..."
Profile:
- doc_type: business_report
- doc_type_description: "Quarterly financial performance report with revenue metrics and business analysis"
- technical_density: 0.5
- reading_level: intermediate
- domain_tags: ["finance", "business", "quarterly_results"]
- best_retrieval_strategy: hybrid
- strategy_confidence: 0.75
- has_math: False
- has_code: False
- summary: "Q3 financial report showing strong revenue growth and key business metrics. Highlights enterprise segment performance and operational efficiency improvements."
- key_concepts: ["revenue_growth", "ARR", "gross_margin", "customer_acquisition", "enterprise_segment"]
Reasoning: Mix of exact metrics (need keyword search for "$145M", "23% YoY") and conceptual analysis (need semantic for "enterprise growth drivers"). Hybrid strategy optimal for both specific numbers and business insights.
</few_shot_examples>

<classification_criteria>
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
</classification_criteria>

<important_instructions>
- Be SPECIFIC with doc_type (choose conference_paper over research_paper if it's from a conference)
- Always provide doc_type_description with meaningful clarification
- Consider the ENTIRE document context, not just the beginning
- Domain tags should be specific and relevant (avoid generic tags like "general" unless truly general)
- If document is truly novel or unique, use "other" and describe it well
- Follow the reasoning patterns shown in the few-shot examples above
</important_instructions>
"""

        try:
            profile = self.structured_llm.invoke([HumanMessage(content=prompt)])
            return profile
        except Exception as e:
            # Fallback to safe defaults if LLM call fails
            print(f"Warning: LLM profiling failed for {doc_id}: {e}")
            return self._get_fallback_profile(doc_text, doc_id)

    def _get_fallback_profile(self, doc_text: str, doc_id: str = None) -> DocumentProfile:
        """Generate fallback profile if LLM call fails (uses simple heuristics as safety net)."""
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
        """Profile multiple documents, returning dict mapping doc_id to DocumentProfile."""
        profiles = {}
        for doc_id, doc_text in documents:
            profiles[doc_id] = self.profile_document(doc_text, doc_id)
        return profiles
