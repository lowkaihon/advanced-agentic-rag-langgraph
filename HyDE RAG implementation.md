# HyDE (Hypothetical Document Embeddings) RAG implementation: What are the best practices for prompting an LLM to generate

HyDE (Hypothetical Document Embeddings) is a powerful technique that flips the retrieval problem: instead of matching a short, ambiguous query to long documents, you generate a "fake" (hypothetical) answer and match that answer to real documents in your database.

The quality of your retrieval depends entirely on the **relevance patterns** (keywords, semantic structure, and jargon) present in that hypothetical document. The prompt is the engine that ensures these patterns align with your ground-truth corpus.

### **Executive Summary: The Core Heuristic**

To prompt for HyDE effectively, you must optimize for **semantic density**, not factual accuracy. The generated document does not need to be true; it needs to look, sound, and read exactly like the valid documents in your vector database.

* **Goal:** Generate text that maximizes cosine similarity with your target retrieval corpus.
* **Key Constraint:** Keep the output within your embedding model's token limit (usually 512 tokens).
* **Best Practice:** Enforce the *style* of the target corpus (e.g., if searching a legal database, prompt the LLM to write in "formal legalese").

***

### **1. Best Practices for HyDE Prompting**

#### **A. Mirror the Target Corpus Style**

The embedding space is sensitive to style and tone. A query in casual English ("how do I fix the memory leak?") might not match a technical documentation corpus as well as a hypothetical snippet written in formal technical language.

* **If searching Reddit:** Prompt for casual, first-person anecdotes.
* **If searching Medical Journals:** Prompt for clinical, passive-voice abstracts using Latin terminology.
* **If searching Code:** Prompt for a code snippet with comments, not a prose explanation.


#### **B. Respect the "Embedding Window"**

Most dense retrievers (like `Contriever`, `all-MiniLM-L6-v2`, or older OpenAI embeddings) have an optimal window of attention.

* **The Trap:** Generating a 1,000-word essay. Most embedding models truncate input after **512 tokens** (approx. 300–400 words).
* **The Fix:** Explicitly constrain the output to a "passage" or "paragraph" (100–300 words). If the key relevance terms appear at the end of a long generation, they will be cut off and ignored by the embedder.


#### **C. Eliminate "Bot Speak"**

Standard LLM pleasantries dilute the semantic signal.

* **Avoid:** "Sure, here is a hypothetical answer..." or "As an AI language model..."
* **Why:** These phrases add noise to the vector embedding, pulling it toward a generic "conversation" cluster rather than your specific topic cluster.
* **Fix:** Use negative constraints in the system prompt to force direct answers.

***

### **2. Recommended Prompt Templates**

Use these templates as starting points. The `{query}` variable represents the user's input.

#### **Scenario A: General Knowledge \& Fact Retrieval**

*Best for: Wikipedia, corporate wikis, general Q\&A.*

```text
PROMPT TEMPLATE:
Please write a concise, informative passage that answers the following question. 
The passage should be written in the style of a formal encyclopedia entry or textbook.
Focus on including key terms, definitions, and factual details.

Question: {query}

Passage:
```


#### **Scenario B: Domain-Specific (e.g., Legal/Finance)**

*Best for: Contracts, case law, financial reports. Note the request for specific "jargon".*

```text
PROMPT TEMPLATE:
Act as an expert legal analyst. Write a hypothetical clause or case summary that is relevant to the query below. 
Use professional legal terminology, citation formats, and formal sentence structures typical of court filings. 
Do not provide a conversational answer; provide a document excerpt.

Query: {query}

Hypothetical Excerpt:
```


#### **Scenario C: Technical Support \& Troubleshooting**

*Best for: IT knowledge bases, Jira tickets, manuals.*

```text
PROMPT TEMPLATE:
You are a senior site reliability engineer. Write a technical troubleshooting step or log entry that addresses the issue described below.
Include specific error codes, command line snippets (in markdown), and configuration parameters that are likely involved.

Issue: {query}

Technical Log Entry:
```


#### **Scenario D: Code Retrieval**

*Best for: Searching GitHub repos or internal codebases.*

```text
PROMPT TEMPLATE:
Write a Python [or specific language] code snippet that solves the following problem. 
Include standard library imports and docstrings. Do not explain the code in text; just provide the code block.

Problem: {query}

Code:
```


***

### **3. System Prompts \& Configuration**

The System Prompt sets the "physics" of the generation. Use this to enforce the length and tone constraints globally.

**Recommended System Prompt:**

```text
You are a HyDE (Hypothetical Document Embeddings) generator. 
Your task is to generate a hypothetical document that contains the semantic patterns, keywords, and technical vocabulary relevant to the user's query.
1. Do NOT answer the user directly. Generate a document that *contains* the answer.
2. Do NOT use conversational fillers ("Here is the answer", "I think").
3. Keep the output between 100-300 words.
4. If the query is ambiguous, generate a passage that covers the most likely interpretation.
```

**Parameter Settings:**

* **Temperature:** `0.7` - `0.9`. Unlike standard RAG (where you want low temp for facts), HyDE benefits from slightly higher creativity to hallucinate plausible keywords that might appear in the target docs.
* **Max Tokens:** `300` (Strict limit to prevent embedding truncation).

***

### **4. Advanced Strategy: Multi-Hypothesis Generation**

If a user query is ambiguous (e.g., "Apple stock"), generating a single hypothetical document might bias the search toward one interpretation (fruit vs. tech company).

**The "N-Voting" Technique:**

1. Generate **N=4** hypothetical documents using the same prompt.
2. Embed all 4 documents.
3. Average the 4 vectors into a single query vector **OR** perform 4 separate searches and rerank the combined results.

**Prompt for N-Voting:**

```text
Write 3 distinct passages that answer the following question from different perspectives or contexts.
Separate each passage with "###".

Question: {query}
```


### **Summary Checklist**

| Component | Recommendation | Reason |
| :-- | :-- | :-- |
| **Length** | 150–300 words | Fits within BERT/MiniLM 512-token limit. |
| **Tone** | Formal/Specific | Matches the "texture" of database documents. |
| **Meta-talk** | Strictly Forbidden | "Sure, here is..." adds noise to the vector. |
| **Truthfulness** | Irrelevant | We need *keywords*, not facts. Hallucination is a feature here. |

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_3][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://machinelearningmastery.com/prompt-engineering-patterns-successful-rag-implementations/

[^1_2]: https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples

[^1_3]: https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py

[^1_4]: https://aclanthology.org/2023.acl-long.99/

[^1_5]: https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api

[^1_6]: https://www.codecademy.com/article/getting-started-with-lang-chain-prompt-templates

[^1_7]: https://docs.llamaindex.ai/en/v0.10.17/api_reference/prompts.html

[^1_8]: https://arxiv.org/abs/2212.10496

[^1_9]: https://www.datacamp.com/tutorial/how-to-improve-rag-performance-5-key-techniques-with-examples

[^1_10]: https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/

[^1_11]: https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings

[^1_12]: https://coralogix.com/ai-blog/enhancing-rag-performance-using-hypothetical-document-embeddings-hyde/

[^1_13]: https://www.sandgarden.com/learn/hyde-embeddings

[^1_14]: https://www.chitika.com/hyde-query-expansion-rag/

[^1_15]: https://www.langchain.asia/modules/chains/index_examples/hyde

[^1_16]: https://medium.aiplanet.com/advanced-rag-improving-retrieval-using-hypothetical-document-embeddings-hyde-1421a8ec075a

[^1_17]: https://www.nature.com/articles/s41746-024-01377-1

[^1_18]: https://pub.towardsai.net/using-hyde-and-reranking-with-qdrant-query-api-to-build-advanced-rag-for-enterprises-9c60d1ae8d4a

[^1_19]: https://arxiv.org/html/2504.14175v1

[^1_20]: https://python.langchain.com/api_reference/_modules/langchain_community/retrievers/web_research.html

[^1_21]: https://haystack.deepset.ai/cookbook/using_hyde_for_improved_retrieval

[^1_22]: https://arxiv.org/abs/2309.02962

[^1_23]: https://news.ycombinator.com/item?id=35841781

[^1_24]: https://milvus.io/docs/full_text_search_with_langchain.md

[^1_25]: https://arxiv.org/html/2406.19760v1

[^1_26]: https://www.wollenlabs.com/blog-posts/what-sentence-embeddings-really-store-and-how-rag-vs-hyde-leverage-them

[^1_27]: https://docs.langchain.com/oss/python/integrations/document_loaders/source_code

