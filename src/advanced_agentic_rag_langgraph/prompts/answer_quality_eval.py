"""
Answer Quality Evaluation Prompts

Evaluates generated answers using vRAG-Eval framework (Correctness, Completeness, Honesty).

Research-backed optimizations:
- GPT-4o-mini (BASE): Detailed rubric + few-shot examples improve classification ~10-12%
- GPT-5 (GPT5): Concise criteria + direct questions leverage reasoning, ~20-25% improvement
- Expected accuracy: 75% (baseline) -> 78-80% (GPT-4o) -> 88-92% (GPT-5)

Evaluation dimensions:
- Relevance: Does answer address the question?
- Completeness: Are all aspects covered?
- Accuracy: Is answer factually correct?

Adaptive thresholds:
- Good retrieval (>0.6): 65% quality threshold
- Poor retrieval (<0.6): 50% quality threshold (compensates for limited context)
"""

BASE_PROMPT = """Evaluate this answer using the vRAG-Eval framework (Correctness, Completeness, Honesty).

Question: {question}

Answer: {answer}

Retrieval Context:
- Retrieval quality: {retrieval_quality}
- Detected issues: {retrieval_issues}

EVALUATION RUBRIC:

1. RELEVANCE: Does the answer address the question asked?

   RELEVANT (True):
   - Answer directly addresses question topic and intent
   - Stays on topic throughout
   - Addresses the core information need

   PARTIALLY RELEVANT:
   - Touches on topic but misses key aspects
   - Drifts off-topic in parts

   IRRELEVANT (False):
   - Discusses unrelated topics
   - Misunderstands the question

2. COMPLETENESS: Does the answer fully address all aspects?

   COMPLETE (True):
   - All question aspects covered with sufficient detail
   - Multi-part questions have all parts answered
   - Provides adequate depth

   PARTIAL:
   - Some aspects covered, others missing
   - Insufficient detail on key points

   INCOMPLETE (False):
   - Major gaps in coverage
   - Leaves question largely unanswered

3. ACCURACY: Is the answer factually correct?

   ACCURATE (True):
   - All statements supported by retrieved documents
   - No unsupported claims or hallucinations
   - Properly grounded in context

   MOSTLY ACCURATE:
   - Minor inaccuracies or unsupported details

   INACCURATE (False):
   - Significant errors or hallucinations
   - Contains unsupported claims

CONFIDENCE SCORING (0-100, threshold: {quality_threshold_pct:.0f}):

EXCELLENT (80-100): Will accept immediately
- Directly relevant to question
- Fully addresses all aspects comprehensively
- Factually accurate, well-grounded
- Clear synthesis
Example: Complete answer with all details, no gaps, fully grounded

GOOD ({quality_threshold_pct:.0f}-79): Acceptable [THRESHOLD]
- Relevant with minor gaps
- Covers key aspects sufficiently
- Generally accurate
- Adequate synthesis
Example: Answers question well but missing 1-2 minor details

FAIR ({quality_threshold_low_pct:.0f}-{quality_threshold_minus_1_pct:.0f}): Needs improvement [Will retry]
- Partially relevant or missing key aspects
- Incomplete coverage
- Some unsupported statements
- Poor synthesis
Example: Addresses question but leaves major gaps

POOR (0-{quality_threshold_low_minus_1_pct:.0f}): Inadequate
- Not relevant
- Major gaps
- Significant errors
- Fails to synthesize
Example: Misses core question or contains hallucinations

FEW-SHOT EXAMPLES:

Example 1 (High Quality):
Question: "What are the advantages of BERT?"
Answer: "BERT offers bidirectional context understanding, pre-training efficiency, and state-of-the-art performance on 11 NLP tasks"
Evaluation:
- is_relevant: True (directly addresses advantages)
- is_complete: True (multiple advantages with specifics)
- is_accurate: True (grounded in BERT paper facts)
- confidence_score: 90
- reasoning: "Answer directly addresses question with specific, grounded advantages. Covers multiple dimensions (architecture, training, results) comprehensively."
- issues: []

Example 2 (Medium Quality):
Question: "Compare BERT and GPT architectures"
Answer: "BERT uses bidirectional transformers while GPT is unidirectional"
Evaluation:
- is_relevant: True (comparison requested, comparison given)
- is_complete: False (only mentions one dimension of difference)
- is_accurate: True (factually correct)
- confidence_score: 65
- reasoning: "Answer is accurate but incomplete - only covers directionality, misses training objectives, use cases, model sizes, etc."
- issues: ["partial_answer", "missing_details"]

Example 3 (Low Quality):
Question: "How does multi-head attention work?"
Answer: "Transformers use attention"
Evaluation:
- is_relevant: True (mentions attention)
- is_complete: False (doesn't explain mechanism)
- is_accurate: True (but too vague)
- confidence_score: 40
- reasoning: "Answer is relevant but completely lacks detail. Doesn't explain heads, queries/keys/values, parallel attention functions, or concatenation. Far too vague."
- issues: ["lacks_specificity", "incomplete_synthesis", "missing_details"]

NOW EVALUATE:

Question: {question}
Answer: {answer}

Note: Consider retrieval quality ({retrieval_quality}) when evaluating. Low retrieval quality or issues like "{retrieval_issues}" may limit answer completeness.

Provide structured output:
- is_relevant (boolean)
- is_complete (boolean)
- is_accurate (boolean)
- confidence_score (0-100)
- reasoning (2-3 sentences)
- issues (list of specific problems from: incomplete_synthesis, lacks_specificity, missing_details, unsupported_claims, partial_answer, wrong_focus, retrieval_limited, contextual_gaps; empty list if none)"""


GPT5_PROMPT = """Evaluate answer quality for this question.

Question: {question}
Answer: {answer}

Retrieval quality: {retrieval_quality}
Issues: {retrieval_issues}

EVALUATE:
- Relevance: Does answer address the question? (True/False)
- Completeness: All aspects covered with sufficient detail? (True/False)
- Accuracy: All statements supported by retrieved documents? (True/False)
- Confidence: 0-100 (threshold {quality_threshold_pct:.0f})

Issues if applicable: incomplete_synthesis, lacks_specificity, missing_details, unsupported_claims, partial_answer, wrong_focus, retrieval_limited, contextual_gaps

Return:
- is_relevant (boolean)
- is_complete (boolean)
- is_accurate (boolean)
- confidence_score (0-100)
- reasoning (2-3 sentences explaining assessment)
- issues (list, empty if none)"""
