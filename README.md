# Advanced Agentic RAG using LangGraph

This Advanced Agentic RAG uses LangGraph to implement features including multi-strategy retrieval (semantic + keyword), LLM-based reranking, intelligent query expansion and rewriting, automatic strategy switching, and self-correcting agent loops with quality evaluation.

## Features

1. Query Optimization
    - Expansion: Generates 3 query variations
    - Rewriting: Rewrites unclear queries if retrieval quality is poor
    - Makes queries clearer and more searchable
2. Intelligent Retrieval
    - Strategy Selection: Picks between semantic, keyword, or hybrid
    - Hybrid Search: Combines BM25 + embeddings
    - Reranking: LLM-as-Judge scores relevance
3. Quality Evaluation
    - Retrieval Quality Score: Rates if documents are relevant
    - Answer Evaluation: Checks if answer meets quality threshold
    - Confidence Scoring: Based on both retrieval + answer quality
4. Self-Correction Loop
    - Strategy Switching: If answer insufficient, tries different retrieval method
    - Query Rewriting: If retrieval poor, rewrites query
    - Iterative Improvement: Up to 3 retrieval attempts
5. Agentic Reasoning
    - Decision Making: LLM decides if retrieval is needed
    - Strategy Selection: LLM picks best retrieval approach
    - Answer Generation: Tailored based on retrieval quality

## Complete Flow

User Question
    ↓
Query Expansion (3 variations)
    ↓
Strategy Selection (semantic/keyword/hybrid)
    ↓
Hybrid Retrieval (semantic + keyword)
    ↓
Reranking (LLM scores relevance)
    ↓
Quality Check (70%+ quality?)
    ├─ NO → Rewrite Query → Retrieve again
    └─ YES → Generate Answer
    ↓
Answer Generation (with quality context)
    ↓
Evaluation (sufficient quality?)
    ├─ NO & attempts < 3 → Switch strategy → Retrieve
    └─ YES → Return Final Answer