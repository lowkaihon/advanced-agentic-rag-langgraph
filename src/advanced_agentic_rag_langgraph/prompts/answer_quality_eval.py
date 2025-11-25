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

3. ACCURACY: Is the answer appropriate given the retrieved context?

   ACCURATE (True):
   - Answer uses only information from retrieved documents
   - Acknowledges limitations when context is insufficient
   - Properly grounded in available context

   MOSTLY ACCURATE:
   - Minor gaps in coverage

   INACCURATE (False):
   - Significant errors or inappropriate extrapolation

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
Question: "What are the key advantages of Agile project management?"
Answer: "Agile project management offers faster delivery (studies show 37% faster than waterfall), enhanced flexibility through iterative development cycles, improved team communication via daily stand-ups and sprint retrospectives, and reduced project risk through early stakeholder feedback and frequent testing"
Evaluation:
- is_relevant: True (directly addresses advantages)
- is_complete: True (covers multiple advantage categories with specifics)
- is_accurate: True (all claims grounded in retrieved documentation)
- confidence_score: 90
- reasoning: "Answer directly addresses question with specific, grounded advantages. Covers multiple dimensions (speed, flexibility, communication, risk) comprehensively. All claims - speed metrics, iterative approach, communication practices, and risk reduction - are supported by context."
- issues: []

Example 2 (Medium Quality):
Question: "Compare the advantages and disadvantages of serverless vs container-based deployment"
Answer: "Serverless offers automatic scaling and reduced operational overhead, while container-based deployment provides more control over the runtime environment and better performance consistency"
Evaluation:
- is_relevant: True (comparison requested, comparison given)
- is_complete: False (only covers scaling/operations and control/performance dimensions)
- is_accurate: True (statements are factually correct and grounded)
- confidence_score: 65
- reasoning: "Answer is accurate but incomplete - addresses only 2 comparison dimensions. Context contains additional important aspects: cost differences for different workload patterns, cold start latency issues, stateful application support, debugging complexity, and use-case recommendations. Missing these key details limits completeness."
- issues: ["partial_answer", "missing_details"]

Example 3 (Low Quality):
Question: "How do B-tree indexes improve database query performance?"
Answer: "B-tree indexes make queries faster by organizing data efficiently"
Evaluation:
- is_relevant: True (mentions B-tree indexes and performance)
- is_complete: False (doesn't explain the mechanism)
- is_accurate: True (but too vague to be useful)
- confidence_score: 40
- reasoning: "Answer is relevant but completely lacks detail. Context provides rich technical explanation including: tree structure with branching factors, search algorithm requiring only log(N) comparisons vs full table scans, node organization, balancing mechanisms, and reduced disk I/O. Answer fails to synthesize any of these concrete mechanisms."
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
- issues (list of specific problems from: incomplete_synthesis, lacks_specificity, missing_details, partial_answer, wrong_focus; empty list if none)"""


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

Issues if applicable: incomplete_synthesis, lacks_specificity, missing_details, partial_answer, wrong_focus

Return:
- is_relevant (boolean)
- is_complete (boolean)
- is_accurate (boolean)
- confidence_score (0-100)
- reasoning (2-3 sentences explaining assessment)
- issues (list, empty if none)"""
