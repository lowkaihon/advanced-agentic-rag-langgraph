## Best Practices for Evaluating Retrieved RAG Documents

Evaluating retrieval quality in RAG systems is crucial for ensuring accurate, relevant, and trustworthy outputs. The evaluation process should assess both retrieval effectiveness and generation quality through a combination of automated metrics, human review, and continuous monitoring. This comprehensive approach enables systematic optimization of RAG pipelines while identifying and addressing specific performance bottlenecks.

### Core Retrieval Evaluation Metrics

**Binary Relevance Metrics**

The foundational metrics for retrieval evaluation operate on binary relevance judgments—whether a retrieved document is relevant or not:[^1_1][^1_2]

**Hit Rate@K** represents the simplest measure, indicating whether at least one relevant document appears in the top K results. It returns 1 if any relevant document exists in the retrieved set, and 0 otherwise.[^1_1]

**Recall@K** measures how many of all relevant documents appear within the top K retrieved results. This metric answers "Out of all existing relevant items, how many did we capture?" and is critical because if relevant documents aren't retrieved initially, no downstream reranking or LLM optimization can rectify the issue.[^1_3][^1_1]

**Precision@K** evaluates how many of the top K retrieved documents are actually relevant, answering "Out of the items we retrieved, how many are correct?" This metric prioritizes quality over quantity and is suitable when retrieving highly confident results matters more than exhaustive retrieval.[^1_3][^1_1]

**F1@K** provides a balanced measure by computing the harmonic mean of Precision@K and Recall@K. An F1 score close to 1 indicates both metrics are high, suggesting retrieved results are both accurate and comprehensive.[^1_1]

**Graded Relevance Metrics**

Beyond binary classification, graded metrics assess varying degrees of relevance and ranking quality:[^1_2][^1_4]

**Normalized Discounted Cumulative Gain (nDCG)** evaluates ranking quality by considering both document relevance and position. More relevant documents appearing earlier in results yield higher scores, with lower-ranked items discounted logarithmically. nDCG is calculated by computing Discounted Cumulative Gain (DCG) and normalizing it against the ideal ranking (IDCG):[^1_4][^1_5][^1_6][^1_7][^1_8]

\$ DCG@K = \sum_{i=1}^{K} \frac{rel(i)}{\log_2(i+1)} \$

\$ nDCG@K = \frac{DCG@K}{IDCG@K} \$

This metric is particularly valuable for RAG systems because the LLM's output quality often depends on retrieval order—critical documents appearing lower in the list may be overlooked or underweighted by the generation model.[^1_6][^1_7]

**Mean Reciprocal Rank (MRR)** focuses on the position of the first relevant document, making it ideal for scenarios where finding the single most relevant result quickly is paramount. However, MRR only handles binary relevance and disregards ranking quality after the first relevant item.[^1_7][^1_9]

### RAG-Specific Evaluation Framework

**Context-Level Metrics**

RAG evaluation extends beyond traditional retrieval metrics to assess how well retrieved context supports generation:[^1_10][^1_11][^1_12]

**Context Relevance (Context Precision)** measures whether retrieved passages contain information genuinely relevant to answering the query, without including extraneous or irrelevant details. Values range from 0-1, with higher scores indicating better relevance. This can be evaluated by having judges assess each retrieved document's relevance and calculating the ratio of relevant to total retrieved documents.[^1_11][^1_12][^1_13][^1_10]

**Context Recall** evaluates how completely the retrieved context matches the ground truth answer or contains all information necessary to produce the ideal output. This metric requires comparison with gold standard answers and indicates whether the retrieval system captured sufficient information.[^1_12][^1_14][^1_11]

**Context Sufficiency** represents the extent to which fetched context includes adequate information to respond correctly to queries, closely related to context recall but emphasizing completeness.[^1_14]

**Contextual Precision (Ranking-Aware)** assesses whether relevant documents are ranked higher than irrelevant ones in the retrieval list, focusing on the re-ranker's effectiveness in ordering results by relevance.[^1_15][^1_16]

**Generation Quality Metrics**

After retrieval, generation metrics assess the LLM's output quality:[^1_13][^1_10][^1_11]

**Faithfulness (Groundedness)** verifies that generated responses are factually grounded in the retrieved context, measuring the fraction of claims in the answer supported by provided context. This metric is essential for detecting hallucinations.[^1_17][^1_18][^1_19][^1_10][^1_11]

**Answer Relevance** measures whether the response actually addresses the user's question. A factually accurate answer discussing the wrong topic scores poorly on relevance despite being correct.[^1_13][^1_17]

**Answer Correctness** evaluates whether the generated answer is factually accurate and includes all necessary information compared to gold standard responses.[^1_14]

### LLM-as-a-Judge Evaluation

Using LLMs to evaluate RAG outputs has become standard practice, offering scalable assessment without extensive ground truth datasets:[^1_20][^1_21][^1_17]

**The RAG Triad Framework** employs an evaluator LLM to assess three dimensions:[^1_21][^1_22][^1_17]

1. **Context Relevance**: Whether retrieved documents align with the user's query
2. **Groundedness**: Whether the generated response accurately reflects retrieved context
3. **Answer Relevance**: Whether the final response addresses the user's original query

The judge LLM generates explanations and scores (typically numerical, binary, or qualitative) for each dimension. Implementation involves creating structured evaluation prompts with specific grading criteria and using the LLM to score both retrieval and generation quality.[^1_17][^1_20][^1_21]

**Advanced Techniques** include pairwise comparison (presenting two responses and selecting the better one), reference-free evaluation (assessing based on predefined criteria), and reference-based evaluation (comparing against verified documents). Tools like DeepEval, RAGAS, and TruLens provide built-in LLM-as-judge evaluators with customizable criteria.[^1_23][^1_22][^1_24][^1_20][^1_15]

**Limitations and Considerations**: LLM-as-judge approaches are sensitive to evaluation prompts and can show inconsistency. Methods like Judge-Consistency (ConsJudge) enhance accuracy by prompting LLMs to generate different judgments based on various dimension combinations and using consistency to select accepted judgments for training. Benchmarking shows that advanced models like GPT-4 and specialized frameworks like FaithJudge achieve 80%+ agreement with human annotations on hallucination detection.[^1_25][^1_18][^1_26][^1_21]

### Building Evaluation Datasets

**Golden Dataset Creation**

High-quality evaluation datasets are foundational for reliable assessment:[^1_27][^1_28][^1_29]

**Components** of a comprehensive test dataset include:[^1_30][^1_27]

- Questions based on source documents
- Ground truth answers (expected accurate responses)
- Retrieved context from the RAG pipeline for each query
- Generated answers from the RAG system

**Synthetic Dataset Generation**: Tools like RAGAS can automatically generate test sets from document corpora, producing relevant queries, expected answers, and ground truth context grounded in specific documents. The process involves creating knowledge graphs from documents, applying transformations to enrich them, and generating scenarios with controlled query distributions.[^1_28][^1_27]

**Manual Curation**: For domain-specific applications, combining synthetic generation with human review ensures test cases reflect real-world complexity and edge cases. Start with small test samples and steadily build toward comprehensive coverage.[^1_29][^1_31][^1_28]

**Continuous Improvement**: Leverage production failures and user interactions to augment golden datasets, making them more representative of actual challenges encountered in deployment.[^1_32][^1_33]

### Retrieval Optimization Strategies

**Embedding Model Selection and Fine-tuning**

Embedding quality directly impacts retrieval performance:[^1_34][^1_35][^1_3]

**Domain-Specific Embeddings**: Models trained on domain-specific data (e.g., PubMedBERT for medical, legal embeddings for law) substantially improve retrieval quality by capturing specialized terminology more effectively. Fine-tuning embedding models on enterprise datasets can yield major improvements in Recall@10 and downstream RAG accuracy.[^1_34][^1_3]

**Evaluation Approach**: Create benchmark datasets mirroring real-world use cases with sample queries paired with known relevant documents. Calculate standard metrics like Recall@K and Precision@K across multiple embedding models, maintaining consistent parameters (e.g., K=5 or K=10). Consider hybrid approaches like reranking initial results with cross-encoders to further improve accuracy.[^1_35]

**Similarity Metrics**: Choose appropriate similarity functions (cosine similarity, dot product, Euclidean distance) based on your embedding model and use case.[^1_30][^1_3]

**Chunking Strategy Optimization**

Chunking decisions critically affect retrieval quality:[^1_36][^1_37][^1_38]

**Chunk Size Considerations**: A common baseline is 512 tokens with 50-100 token overlap, though optimal size varies by content type. Research shows page-level chunking often achieves highest average accuracy (64.8%) with lowest standard deviation, providing consistent performance across diverse datasets. However, specific domains may benefit from different approaches—financial documents showed varying optimal strategies ranging from 512 to 1024 tokens depending on the dataset.[^1_37][^1_36]

**Overlap Strategy**: Typical overlap of 10-20% of chunk size preserves context that might be lost at boundaries. This ensures semantic continuity across chunks.[^1_37]

**Avoid Extremes**: Very small chunks (128 tokens) and very large chunks (2048 tokens) typically underperform, suggesting a "sweet spot" in the middle range for most document types.[^1_36][^1_37]

**Evaluation Process**: Start with a baseline strategy, experiment with different chunk sizes and overlap parameters, and measure impact on retrieval metrics and end-to-end RAG accuracy. Use tools that support chunk visualization to inspect how documents are segmented.[^1_38][^1_36]

**Reranking with Cross-Encoders**

Implementing reranking significantly improves retrieval relevance:[^1_39][^1_40][^1_41][^1_42]

**Architecture**: Cross-encoders process query and document together by concatenating them ([CLS] query [SEP] document [SEP]) and processing through a transformer model to generate relevance scores. This enables rich interaction analysis that bi-encoders cannot capture.[^1_40][^1_43][^1_39]

**Performance Impact**: Cross-encoder reranking improves RAG accuracy by 20-35% but adds 200-500ms latency per query. The typical workflow retrieves 20-50 candidates with fast bi-encoder search, then reranks the top results using a cross-encoder to balance efficiency and accuracy.[^1_43][^1_39][^1_40]

**Model Selection**: Popular models include Cohere Rerank and ms-marco-MiniLM-L-6-v2, which offer strong balance of accuracy and speed for most applications. Rerank the top 5-10 documents for the LLM to maximize relevance while controlling costs.[^1_40]

**Best Practices**: Implement batched reranking for efficiency, use adaptive reranking (deciding when to rerank based on initial score distribution), and consider hybrid approaches combining multiple reranking methods.[^1_43][^1_40]

### Evaluation Frameworks and Tools

**Popular Open-Source Frameworks**

Several specialized frameworks streamline RAG evaluation:[^1_22][^1_44][^1_24][^1_45]

**RAGAS (Retrieval-Augmented Generation Assessment Suite)** provides research-backed metrics including faithfulness, contextual relevancy, answer relevancy, contextual recall, and contextual precision. It offers simple interfaces for creating evaluation datasets and quickly obtaining comprehensive scores. RAGAS allows separate evaluation of retriever and generator components, helping precisely identify pipeline bottlenecks.[^1_24][^1_46][^1_22]

**TruLens-Eval** offers integrated evaluation for LangChain and LlamaIndex-based applications, with visual monitoring in browsers for analyzing evaluation reasons and API usage. It implements feedback functions that execute after each model call to assess dimensions like factuality, coherence, and relevance.[^1_22][^1_24]

**DeepEval** provides RAG-specific metrics with LLM-as-judge scorers, supporting component-level tracing with the @observe decorator. It enables evaluation of nested RAG pipelines and offers golden dataset management with EvaluationDataset classes.[^1_47][^1_19]

**LangSmith** has emerged as the enterprise standard with tight LangGraph integration enabling production-grade observability, debugging, and evaluation. It allows end-to-end tracing of retrieval and generation flows with real-time insights at scale.[^1_44]

**Framework Selection Criteria**: Choose based on your stack compatibility (LangChain, LlamaIndex, custom), evaluation goals (retrieval accuracy vs. faithfulness vs. ethical considerations), and whether you need research flexibility or production monitoring.[^1_45]

### Production Evaluation Strategies

**Offline vs. Online Evaluation**

Effective RAG systems require both evaluation approaches working in tandem:[^1_48][^1_49][^1_32]

**Offline Evaluation** occurs in controlled environments using golden datasets to assess system changes before production deployment:[^1_32]

- **Use Cases**: Pre-deployment testing, regression testing, component-level optimization, comparative analysis of different configurations[^1_32]
- **Methodology**: Curate representative golden datasets, apply comprehensive metrics (RAGAS, faithfulness, answer relevancy), conduct A/B comparisons under identical conditions[^1_32]
- **Advantages**: Controlled environment, reproducible results, safe testing of experimental changes, enables thorough analysis with computationally intensive metrics[^1_32]
- **Limitations**: Dataset dependency, potential disconnect from real user behavior, static nature misses dynamic effects[^1_32]

**Online Evaluation** monitors live system performance with real user interactions:[^1_49][^1_33][^1_32]

- **Use Cases**: Continuous production monitoring, A/B testing, validating offline findings, detecting drift and evolving patterns[^1_33][^1_32]
- **Methodology**: Collect implicit feedback (click-through rates, session duration, task completion), explicit feedback (ratings, thumbs up/down), conduct A/B tests comparing system versions[^1_32]
- **Advantages**: Measures real user experience, detects live issues immediately, captures dynamic effects and data drift[^1_32]
- **Limitations**: Requires substantial traffic, potential negative user impact during testing, privacy considerations, noisier signals than controlled tests[^1_32]

**Integration Strategy**: Create feedback loops where online insights augment offline datasets with frequently problematic queries and edge cases, while offline analysis generates hypotheses tested through online A/B experiments. Implement comprehensive logging, evaluation frameworks, A/B testing platforms, and monitoring systems to support both approaches.[^1_33][^1_32]

**Human-in-the-Loop Evaluation**

Incorporating human judgment addresses gaps in automated metrics:[^1_50][^1_51][^1_52][^1_53]

**Complementary Role**: While automated metrics provide scalable quantitative scores, they often miss subjective qualities like clarity, correctness, coherence, and usefulness. Human evaluators assess dimensions that are algorithmically difficult to quantify, such as logical structure, factual accuracy, and appropriateness for user intent.[^1_52]

**Critical Use Cases**: Identifying edge cases where automated metrics fail (e.g., technically correct but overly verbose answers), rating qualitative dimensions (1-5 scales for clarity), providing actionable feedback for model refinement.[^1_53][^1_52]

**Structured Workflow**: Package flagged responses with original query, model answer, retrieved documents, and relevant automated metrics (e.g., RAGAS scores) so reviewers have necessary context. This makes human review efficient and scalable.[^1_53]

**Hybrid Approach**: Use automated tools for large-scale testing and obvious error detection, while human judges focus on qualitative improvements and calibrating automated metrics based on domain-specific priorities. This creates a balanced feedback loop that ensures both efficiency and depth of evaluation.[^1_52]

**Continuous Improvement Loop**: Integrate human feedback into training data, use it to refine retrieval signals (indicating which documents are most reliable), and establish adaptive systems that learn from moderation to prioritize pertinent documents for future queries.[^1_51][^1_33]

**Production Monitoring**

Continuous monitoring extends evaluation into operational environments:[^1_54][^1_49][^1_33]

**Trace Logging**: Track inputs, outputs, and intermediate steps (document retrieval) to enable ongoing monitoring and diagnosis of issues. Distributed tracing captures complete execution paths for every interaction.[^1_54][^1_33]

**Automated Evaluations**: Run evaluators continuously against live data with real-time alerts when quality metrics degrade. Configure sampling and filters to control evaluation costs while maintaining coverage.[^1_49][^1_33]

**Drift Detection**: Monitor for changes in data distributions, user query patterns, and seasonal trends that may degrade performance over time.[^1_49][^1_32]

**Alert Configuration**: Create alerts for log metrics (latency, token usage, cost) and evaluation scores (toxicity, bias, clarity, relevance) with integrations to Slack or PagerDuty for immediate notification.[^1_33][^1_49]

### Implementation Best Practices

**Comprehensive Multi-Metric Approach**

Combine multiple evaluation dimensions for holistic assessment:[^1_10][^1_50][^1_30][^1_13]

- Evaluate retrieval (precision, recall, nDCG) and generation (faithfulness, relevance) separately to isolate performance issues
- Use both reference-based metrics (when ground truth exists) and reference-free LLM-as-judge evaluation
- Implement domain-specific custom metrics relevant to your use case[^1_50]

**Iterative Optimization Workflow**

Systematic improvement requires structured processes:[^1_30][^1_36][^1_32]

1. Establish baseline performance with standard configurations (e.g., 512-token chunks, basic retrieval)
2. Implement comprehensive logging and evaluation infrastructure
3. Conduct offline experiments varying one component at a time (embedding model, chunk size, reranking)
4. Validate promising changes through online A/B testing with subset of users
5. Monitor production metrics continuously and feed insights back to golden datasets
6. Repeat cycle for continuous improvement[^1_33][^1_32]

**Error Analysis and Debugging**

Investigate patterns in low-scoring responses to identify root causes:[^1_50][^1_54]

- Analyze whether failures stem from retrieval (missing relevant documents) or generation (hallucination despite good context)
- Examine edge cases and failure modes systematically
- Use component-level metrics to pinpoint specific pipeline bottlenecks
- Build datasets from production failures to strengthen future testing[^1_33]

**Regular Benchmarking**

Continuously evaluate as pipeline changes occur:[^1_48][^1_50]

- Schedule periodic health checks even when no changes are deployed to detect gradual degradation
- Conduct regression testing to ensure updates don't degrade previously established performance
- Compare against baseline models and alternative implementations
- Track performance trends over time to identify patterns

By implementing these comprehensive best practices—combining automated metrics, human evaluation, production monitoring, and continuous optimization—you can build RAG systems that maintain high accuracy, relevance, and reliability throughout their lifecycle while efficiently identifying and addressing performance issues as they emerge.
<span style="display:none">[^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84]</span>

<div align="center">⁂</div>

[^1_1]: https://towardsdatascience.com/how-to-evaluate-retrieval-quality-in-rag-pipelines-precisionk-recallk-and-f1k/

[^1_2]: https://deconvoluteai.com/blog/rag/metrics-retrieval

[^1_3]: https://toloka.ai/blog/rag-evaluation-a-technical-guide-to-measuring-retrieval-augmented-generation/

[^1_4]: https://www.ai21.com/knowledge/rag-evaluation/

[^1_5]: https://www.linkedin.com/pulse/7-retrieval-metrics-better-rag-systems-asgrag-x5fyc

[^1_6]: https://milvus.io/ai-quick-reference/how-can-we-incorporate-metrics-like-ndcg-normalized-discounted-cumulative-gain-to-evaluate-ranked-retrieval-outputs-in-a-rag-context-where-document-order-may-influence-the-generator

[^1_7]: https://www.evidentlyai.com/ranking-metrics/ndcg-metric

[^1_8]: https://weaviate.io/blog/retrieval-evaluation-metrics

[^1_9]: https://blog.stackademic.com/ndcg-vs-mrr-ranking-metrics-for-information-retrieval-in-rags-2061b04298a6

[^1_10]: https://orq.ai/blog/rag-evaluation

[^1_11]: https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more

[^1_12]: https://aws.amazon.com/blogs/machine-learning/evaluate-the-reliability-of-retrieval-augmented-generation-applications-using-amazon-bedrock/

[^1_13]: https://www.braintrust.dev/articles/rag-evaluation-metrics

[^1_14]: https://www.patronus.ai/llm-testing/rag-evaluation-metrics

[^1_15]: https://haystack.deepset.ai/cookbook/rag_eval_deep_eval

[^1_16]: https://deepeval.com/docs/metrics-contextual-precision

[^1_17]: https://mistral.ai/news/llm-as-rag-judge

[^1_18]: https://arxiv.org/html/2505.04847v2

[^1_19]: https://machinelearningmastery.com/rag-hallucination-detection-techniques/

[^1_20]: https://www.nb-data.com/p/evaluating-rag-with-llm-as-a-judge

[^1_21]: https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/

[^1_22]: https://zilliz.com/blog/how-to-evaluate-retrieval-augmented-generation-rag-applications

[^1_23]: https://www.evidentlyai.com/llm-guide/llm-as-a-judge

[^1_24]: https://research.aimultiple.com/llm-eval-tools/

[^1_25]: https://arxiv.org/abs/2502.18817

[^1_26]: https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking/

[^1_27]: https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/

[^1_28]: https://langfuse.com/guides/cookbook/example_synthetic_datasets

[^1_29]: https://www.evidentlyai.com/llm-guide/llm-test-dataset-synthetic-data

[^1_30]: https://qdrant.tech/blog/rag-evaluation-guide/

[^1_31]: https://gentrace.ai/blog/how-to-build-datasets

[^1_32]: https://apxml.com/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/rag-offline-online-evaluation

[^1_33]: https://dev.to/kuldeep_paul/top-5-rag-evaluation-platforms-in-2025-2i0g

[^1_34]: https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning

[^1_35]: https://milvus.io/ai-quick-reference/how-can-we-evaluate-different-embedding-models-to-decide-which-yields-the-best-retrieval-performance-for-our-specific-rag-use-case

[^1_36]: https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

[^1_37]: https://weaviate.io/blog/chunking-strategies-for-rag

[^1_38]: https://www.llamaindex.ai/blog/efficient-chunk-size-optimization-for-rag-pipelines-with-llamacloud

[^1_39]: https://eyka.com/blog/reranking-in-rag-enhancing-accuracy-with-cross-encoders/

[^1_40]: https://customgpt.ai/rag-reranking-techniques/

[^1_41]: https://towardsdatascience.com/rag-explained-reranking-for-better-answers/

[^1_42]: https://www.cloudthat.com/resources/blog/the-power-of-cross-encoders-in-re-ranking-for-nlp-and-rag-systems/

[^1_43]: https://atalupadhyay.wordpress.com/2025/06/19/reranking-in-rag-pipelines-a-complete-guide-with-hands-on-implementation/

[^1_44]: https://www.linkedin.com/posts/ai-with-bhagwat-chate_genai-langchain-langgraph-activity-7377177041802133504-uqms

[^1_45]: https://www.deepchecks.com/best-rag-evaluation-tools/

[^1_46]: https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html

[^1_47]: https://deepeval.com/docs/getting-started-rag

[^1_48]: https://aws.amazon.com/blogs/machine-learning/evaluating-rag-applications-with-amazon-bedrock-knowledge-base-evaluation/

[^1_49]: https://www.getmaxim.ai/articles/rag-evaluation-a-complete-guide-for-2025/

[^1_50]: https://aiengineering.academy/RAG/01_RAG_Evaluation/

[^1_51]: https://www.reddit.com/r/Rag/comments/1g1sgen/is_human_in_the_loop_the_key_to_improving_rag/

[^1_52]: https://milvus.io/ai-quick-reference/in-evaluating-answer-quality-how-can-human-evaluation-complement-automated-metrics-for-rag-eg-judges-rating-clarity-correctness-and-usefulness-of-answers

[^1_53]: https://labelstud.io/blog/why-human-review-is-essential-for-better-rag-systems/

[^1_54]: https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/fundamentals-evaluation-monitoring-rag

[^1_55]: https://www.meilisearch.com/blog/rag-evaluation

[^1_56]: https://arxiv.org/abs/2404.13781

[^1_57]: https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/rag-evaluators

[^1_58]: https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/

[^1_59]: https://arxiv.org/html/2504.14891v1

[^1_60]: https://deepeval.com/guides/guides-rag-evaluation

[^1_61]: https://www.braintrust.dev/articles/best-rag-evaluation-tools

[^1_62]: https://aclanthology.org/2025.coling-main.449.pdf

[^1_63]: https://www.superannotate.com/blog/rag-evaluation

[^1_64]: https://aiexponent.com/the-complete-enterprise-guide-to-rag-evaluation-and-benchmarking/

[^1_65]: https://www.evidentlyai.com/llm-guide/rag-evaluation

[^1_66]: https://stackoverflow.com/questions/79488543/what-to-include-in-context-precision-recall-for-rag-llm-evaluation

[^1_67]: https://aws.amazon.com/blogs/aws/new-rag-evaluation-and-llm-as-a-judge-capabilities-in-amazon-bedrock/

[^1_68]: https://docs.ragas.io/en/v0.1.21/concepts/metrics/

[^1_69]: https://lightning.ai/panchamsnotes/studios/evaluate-your-rag-part-2-llm-as-a-judge

[^1_70]: https://openreview.net/forum?id=hgBZP67BkP

[^1_71]: https://www.confident-ai.com/docs/llm-evaluation/core-concepts/test-cases-goldens-datasets

[^1_72]: https://arxiv.org/abs/2505.04847

[^1_73]: https://www.datadoghq.com/blog/ai/llm-hallucination-detection/

[^1_74]: https://www.reddit.com/r/LangChain/comments/1bijg75/why_is_everyone_using_ragas_for_rag_evaluation/

[^1_75]: https://www.relari.ai/blog/how-important-is-a-golden-dataset-for-llm-evaluation

[^1_76]: https://www.reddit.com/r/Rag/comments/1hcpdke/how_to_find_the_most_optimal_size_for_chunking/

[^1_77]: https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089

[^1_78]: https://www.reddit.com/r/Rag/comments/1kzkoaf/this_paper_eliminates_reranking_in_rag/

[^1_79]: https://www.pinecone.io/learn/chunking-strategies/

[^1_80]: https://docs.ragas.io/en/v0.3.5/howtos/applications/compare_embeddings/

[^1_81]: https://www.pinecone.io/learn/series/rag/rerankers/

[^1_82]: https://milvus.io/ai-quick-reference/what-is-the-difference-between-online-and-offline-evaluation-of-recommender-systems

[^1_83]: https://zilliz.com/ai-faq/what-is-the-difference-between-online-and-offline-evaluation-of-recommender-systems

[^1_84]: https://arxiv.org/html/2508.11062v1

