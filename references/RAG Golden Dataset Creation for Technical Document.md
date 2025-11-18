## RAG Golden Dataset Creation for Technical Documents: Strategies for Research Paper Evaluation

### Foundation: Understanding the Golden Dataset

A **golden dataset** serves as the ground truth benchmark for evaluating RAG systems, consisting of high-quality, human-verified question-document-answer triplets that mirror production use cases. Unlike silver datasets (synthetic or automatically generated), golden datasets provide reliable baselines for measuring retrieval and generation performance, though they require significant investment in curation and expert validation.[^1_1][^1_2]

### Scope Definition and Goal Alignment

The first critical step involves defining precise evaluation scope and objectives. For technical documents and research papers, scope decisions should address:[^1_1]

**Evaluation Units**: Determine whether evaluating at the session level (end-to-end RAG workflow), trace level (individual retrieval steps), or span level (specific claim verification). For research papers, span-level evaluation becomes particularly important when assessing citation grounding and claim attribution.[^1_1]

**Domain-Specific Metrics**: Prioritize metrics that reflect user value specific to technical content. Beyond standard correctness, consider groundedness (whether answers are supported by retrieved sources), completeness (addressing all relevant aspects), and citation precision (accurate attribution to source materials).[^1_3][^1_4][^1_1]

**Task Taxonomy**: Identify the types of queries your system must handle. For research papers, this includes factual questions ("What is the key methodology?"), comparative analysis ("How does this approach differ from prior work?"), and multi-hop reasoning ("Which papers build upon this finding and how?").[^1_5]

### Data Sourcing Strategies

**Production-Representative Scenarios**: Extract real logs from domain users or domain experts, ensuring datasets reflect actual information-seeking patterns. For research papers, this might involve queries from literature review processes, grant writing, or academic research workflows.[^1_1]

**Subject Matter Expert Curation**: Involve domain experts early to author "must-pass" scenarios with explicit acceptance criteria. For technical content, experts should verify that questions are well-formed, answerable from the corpus, and representative of realistic research tasks.[^1_1]

**Citation-Based Expansion**: Leverage scholarly infrastructure. Begin with core documents selected by experts, then expand using APIs like Semantic Scholar, SCOPUS, and OSTI to build interconnected paper collections. Apply automatic embedding-based pruning (using BERT-based models fine-tuned on scientific literature) to maintain thematic coherence.[^1_6]

### Annotation Framework Design

**Clear Annotation Guidelines**: Develop comprehensive guidelines that define relevance criteria, handle ambiguity, and provide positive and negative examples. Guidelines should explain:[^1_7]

- What makes a document relevant to a query (exact topical match, methodological contribution, comparative reference)
- How to handle edge cases (papers that partially address the question, tangential citations)
- Conflict resolution procedures when documents have multiple legitimate relevance levels[^1_7]

**Multi-Level Relevance Judgments**: For research papers, consider graduated relevance levels rather than binary relevant/irrelevant:

- Highly Relevant: Contains direct answers or primary evidence
- Relevant: Contributes context or supporting evidence
- Marginally Relevant: Related but not essential for answering
- Not Relevant: No meaningful connection

This enables richer evaluation through metrics like NDCG@k that reward ranking quality.[^1_8][^1_9]

**Grounding Annotations**: For each question-answer pair, explicitly annotate which passages support each claim in the answer. Research shows that maintaining detailed grounding annotations significantly improves downstream RAG evaluation.[^1_10]

### Inter-Annotator Agreement and Quality Control

Establishing reliability in annotations is paramount. Use standardized measures:[^1_11]

**Cohen's Kappa** (for pairwise agreement on categorical data): Measures agreement corrected for chance. Values above 0.61-0.80 indicate substantial agreement, while 0.81+ indicates almost perfect agreement.[^1_12][^1_11]

**Krippendorff's Alpha** (for multiple annotators): Accounts for sample size, category diversity, and chance agreement, ranging from 0 to 1 where 1 indicates perfect agreement.[^1_11]

**Implementation Strategy**:

1. Start with a pilot annotation round on 50-100 documents with 2-3 domain experts
2. Calculate inter-annotator agreement metrics
3. Review disagreements to identify ambiguous guidelines
4. Refine guidelines and iterate until achieving Krippendorff's Alpha > 0.70[^1_11]
5. Use disagreement instances to identify edge cases for explicit handling[^1_1]

### Dataset Size and Composition

The appropriate dataset size depends on task complexity and available resources:

**Starter Phase**: 50-100 examples for initial evaluation of low-risk systems[^1_13]

**Standard Evaluation**: 200-500 diverse examples covering different complexity levels, question types, and document domains. For research papers, ensure representation across subdomains (computer science, mathematics, physics, etc.)[^1_14][^1_13]

**Comprehensive Benchmarks**: 1000+ examples for production systems where failures have high cost. Large public benchmarks like RAGBench contain 100k examples across five industry domains.[^1_15][^1_13]

**Diversity Dimensions**: Ensure coverage of:

- **Temporal dynamics**: Mix recent papers and foundational works
- **Complexity levels**: Simple factual retrieval to complex multi-hop reasoning
- **Query formulations**: Keywords, natural language questions, structured queries
- **Document types**: Full papers, sections, tables, supplementary materials
- **Edge cases**: Contradictory findings, papers addressing same topic with different conclusions[^1_10]


### Synthetic Data and Silver→Gold Promotion

**Silver Dataset Generation**: Use LLMs to generate synthetic query-document pairs initially. Recent research shows that relative query generation (generating queries relative to other relevance levels rather than in isolation) produces higher-quality synthetic data.[^1_16]

**Constraints-Based Generation**: When generating synthetic queries, incorporate domain-specific constraints. For research papers, prompt LLMs to generate questions at multiple granularity levels (document-level, section-level, sentence-level) and with metadata constraints (e.g., "generate questions about methodology from computer science papers").[^1_17]

**Human-in-the-Loop Promotion**: Establish a clear promotion pathway from silver to gold:

1. Generate diverse synthetic Q\&A pairs at scale
2. Sample strategically for expert review (1-5% of synthetic data)
3. Promote reviewed samples to golden tier
4. Track confidence scores to identify uncertain samples for additional review[^1_1]

### Evaluation Metrics Framework

Implement a **multi-layered metric strategy** addressing retrieval, generation, and end-to-end performance:

**Retrieval-Level Metrics**:[^1_4][^1_9][^1_18][^1_8]

- **Recall@k**: Percentage of relevant documents in top-k results. Critical for research papers where relevant evidence may be distributed
- **Precision@k**: Proportion of top-k results that are relevant
- **Mean Reciprocal Rank (MRR)**: Rewards systems that rank the first relevant document highly. Suitable when one document suffices
- **Mean Average Precision (MAP)**: Considers both presence and ranking of all relevant documents
- **Normalized Discounted Cumulative Gain (nDCG@k)**: Most comprehensive for research papers, rewards relevant documents appearing earlier while accounting for varying relevance degrees

**Generation-Level Metrics**:[^1_18][^1_4]

- **Citation Precision and Recall**: Whether generated answers properly attribute claims to retrieved sources
- **Groundedness Score**: Percentage of claims supported by retrieved passages
- **Answer Relevance**: Whether the answer addresses the query completely
- **Hallucination Rate**: Percentage of unsupported or contradictory claims

**End-to-End Metrics**:[^1_4]

- **Factual Consistency**: Whether final output aligns with all retrieved passages and knowledge base
- **Answer Completeness**: Coverage of all query dimensions


### Production Governance and Monitoring

**Dataset Versioning and Lineage**: Implement versioning for the golden dataset with clear documentation of:

- Data provenance (source, collection method)
- Annotation guidelines version
- Annotator identity and expertise
- Consent flags and privacy considerations
- Risk tags for sensitive content[^1_19][^1_1]

This aligns with ISO/IEC 42001 requirements for AI governance and NIST AI RMF compliance.[^1_20][^1_19]

**Continuous Evolution**: Golden datasets should grow dynamically:

1. Feed production failure cases back to the evaluation set
2. Monitor for distribution shift in user queries
3. Quarterly reviews to incorporate new paper domains or query types
4. Track metric trends to detect performance degradation[^1_20][^1_1]

### Hallucination Detection and Groundedness

For research papers where factual accuracy is critical, implement specialized evaluation:

**Retrieval-Based Hallucination Detection**: Use retrievers with Natural Language Inference (NLI) models to predict factual consistency between generated claims and retrieved context. Recent frameworks achieve F1 scores of 0.83 in detecting unsupported claims.[^1_21]

**Attribution Verification**: Require inline citations with span-level alignment. Implement automated checks that:

- Verify every factual claim traces to specific retrieved passages
- Identify claims marked as general knowledge versus source-grounded
- Flag responses that contradict retrieved evidence[^1_22]


### Error Analysis and Failure Mode Categorization

Systematic error analysis surfaces domain-specific failure patterns:[^1_23]

1. **Identify failure modes** through qualitative review of 50-100 traces:
    - Retrieval failures (wrong documents selected)
    - Hallucinations (incorrect information generation)
    - Multi-hop failures (incorrect connection of evidence)
    - Citation errors (wrong source attribution)
2. **Quantify** frequency and impact of each failure type
3. **Build targeted evaluators** for identified patterns rather than relying solely on generic metrics

For research papers, retrieval failures often stem from vocabulary mismatch (terminology differences across subfields) or insufficient context. Domain-specific embedding models improve performance significantly.[^1_24]

### Implementation Recommendations

**Initial Phase** (100-200 examples):

- Define scope and task taxonomy clearly
- Develop annotation guidelines with 3 expert rounds
- Create a pilot golden set with Krippendorff's Alpha > 0.70
- Establish baseline metrics across all dimensions

**Growth Phase** (500-1000 examples):

- Implement active learning to prioritize high-uncertainty samples from production
- Use LLM-as-judge to scale annotation, but validate on gold standard samples
- Expand coverage to underrepresented query types and document domains

**Production Phase** (1000+ examples):

- Establish continuous monitoring with hallucination detection
- Implement automated regression testing against golden dataset
- Create error analysis dashboards tracking failure modes over time
- Schedule quarterly dataset audits and versioning


### Tool and Framework Considerations

For technical documents and research papers, consider:

- **Hybrid retrieval**: Combine BM25 (lexical) and embedding-based (semantic) retrieval to handle domain terminology and conceptual matching
- **Domain-specific embeddings**: Fine-tune retrievers on scientific literature or your target domain
- **Knowledge graphs**: For research papers with complex citation networks, graph-based approaches can represent multi-hop relationships
- **Multi-modal annotation**: Handle tables, figures, and formulas alongside text

The investment in golden dataset creation for technical documents pays dividends through reliable evaluation, faster iteration cycles, and production-ready systems that users can trust for research and knowledge discovery tasks.
<span style="display:none">[^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56]</span>

<div align="center">⁂</div>

[^1_1]: https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/

[^1_2]: https://jakobs.dev/evaluating-rag-synthetic-dataset-generation/

[^1_3]: https://publicera.kb.se/ir/article/view/40906

[^1_4]: https://neptune.ai/blog/evaluating-rag-pipelines

[^1_5]: https://milvus.io/ai-quick-reference/what-are-some-standard-benchmarks-or-datasets-used-to-test-retrieval-performance-in-rag-systems-for-instance-opendomain-qa-benchmarks-like-natural-questions-or-webquestions

[^1_6]: https://arxiv.org/html/2410.02721v1

[^1_7]: https://tinkogroup.com/how-to-draft-effective-annotation-guidelines/

[^1_8]: https://www.linkedin.com/pulse/7-retrieval-metrics-better-rag-systems-asgrag-x5fyc

[^1_9]: https://weaviate.io/blog/retrieval-evaluation-metrics

[^1_10]: https://aclanthology.org/2025.findings-acl.875.pdf

[^1_11]: https://www.innovatiana.com/en/post/inter-annotator-agreement

[^1_12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/

[^1_13]: https://www.datagrid.com/blog/llm-evaluation-metrics-guide

[^1_14]: https://innodata.com/what-are-golden-datasets-in-ai/

[^1_15]: https://arxiv.org/abs/2407.11005

[^1_16]: https://research.google/pubs/its-all-relative-a-synthetic-query-generation-approach-for-improving-zero-shot-relevance-prediction/

[^1_17]: https://aclanthology.org/2025.acl-long.392.pdf

[^1_18]: https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/

[^1_19]: https://petronellatech.com/blog/from-policy-to-proof-iso-iec-42001-the-os-for-enterprise-ai/

[^1_20]: https://aws.amazon.com/blogs/security/ai-lifecycle-risk-management-iso-iec-420012023-for-ai-governance/

[^1_21]: https://arxiv.org/abs/2504.15771

[^1_22]: https://www.getmaxim.ai/articles/llm-hallucination-detection-and-mitigation-best-techniques/

[^1_23]: https://langfuse.com/blog/2025-08-29-error-analysis-to-evaluate-llm-applications

[^1_24]: https://arxiv.org/pdf/2509.04139.pdf

[^1_25]: https://www.relari.ai/blog/how-important-is-a-golden-dataset-for-llm-evaluation

[^1_26]: https://arxiv.org/abs/2505.08643

[^1_27]: https://arxiv.org/html/2404.13781v1

[^1_28]: https://www.deepset.ai/blog/rag-evaluation-retrieval

[^1_29]: https://milvus.io/ai-quick-reference/why-is-it-important-to-prepare-a-dedicated-evaluation-dataset-for-rag-and-what-should-the-key-components-of-such-a-dataset-be

[^1_30]: https://forage.ai/blog/document-annotation-unlocking-business-intelligence-from-unstructured-data/

[^1_31]: https://ad-publications.cs.uni-freiburg.de/benchmark.pdf

[^1_32]: https://gipplab.uni-goettingen.de/wp-content/papercite-data/pdf/meuschke2023.pdf

[^1_33]: https://www.reddit.com/r/Rag/comments/1m0fxax/rag_system_for_technical_documents_tips/

[^1_34]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4893911/

[^1_35]: https://www.gdpicture.com/blog/the-complete-guide-to-document-annotation/

[^1_36]: https://aclanthology.org/W08-1209.pdf

[^1_37]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-124.pdf

[^1_38]: https://joshpitzalis.com/2025/06/06/llm-evaluation/

[^1_39]: https://www.datadoghq.com/blog/llm-observability-hallucination-detection/

[^1_40]: https://www.schellman.com/blog/ai-services/hitrust-ai-security-assessment-path-to-iso-42001

[^1_41]: https://arxiv.org/html/2406.14891v1

[^1_42]: https://www.semanticscholar.org/faq

[^1_43]: https://www.semanticscholar.org/paper/Semantic-Matching-in-Search-Li-Xu/d158cc9cd09e944a9d275dab8388022a5e1da674

[^1_44]: https://www.superannotate.com/blog/how-to-write-data-annotation-style-guide

[^1_45]: https://qirui-chen.github.io/MultiHop-EgoQA/

[^1_46]: https://library.smu.edu.sg/topics-insights/google-scholar-vs-ai-academic-search-expanding-literature-review-toolkit

[^1_47]: https://arxiv.org/html/2511.07685v1

[^1_48]: https://www.semanticscholar.org/paper/MINTQA:-A-Multi-Hop-Question-Answering-Benchmark-on-He-Hu/de74cdb6348410bd3058c6089160f60b4e4a1667

[^1_49]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4144157/

[^1_50]: https://arize.com/blog/llm-as-judge-survey-paper/

[^1_51]: https://www.sciencedirect.com/science/article/abs/pii/S2214860421005649

[^1_52]: https://sigma.ai/golden-datasets/

[^1_53]: https://arxiv.org/html/2508.02994v1

[^1_54]: https://www.ijcai.org/proceedings/2018/0134.pdf

[^1_55]: https://arxiv.org/abs/2411.15594

[^1_56]: https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.582470/full

