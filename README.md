# Advanced Agentic RAG using LangGraph

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain 1.0](https://img.shields.io/badge/LangChain-1.0-green.svg)
![LangGraph 1.0](https://img.shields.io/badge/LangGraph-1.0-purple.svg)

An advanced Agentic RAG system that autonomously adapts its retrieval strategy and reasoning process through dynamic decision-making, iterative self-correction, and intelligent tool selection. Built with LangGraph's StateGraph pattern, the system embeds autonomous reasoning into a 7-node workflow where routing functions and conditional edges provide distributed intelligence--no central "agent" orchestrator needed.

## Demo

https://github.com/user-attachments/assets/c4168ac9-3eb0-45dc-be67-895299d8a97e

## Key Results

- **2.3x retrieval accuracy** (F1@4: 13.1% -> 29.6%) with budget models only
- Demonstrates architectural value independent of model quality
- [Full evaluation details](#evaluation)

## Table of Contents

- [Why This Qualifies as Agentic RAG](#why-this-qualifies-as-agentic-rag)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Architecture Tiers](#architecture-tiers)
- [Quick Start](#quick-start)
- [Model Tier Configuration](#model-tier-configuration)
- [Technology Stack](#technology-stack)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)

## Why This Qualifies as Agentic RAG

This system demonstrates all four core characteristics that define Agentic RAG:

### 1. Autonomous Decision-Making
**Traditional RAG**: Fixed pipeline (query -> retrieve -> generate)<br>
**This System**: Dynamic routing based on quality evaluation at 2 decision points

The system autonomously plans next steps based on intermediate results:
- `route_after_retrieval`: Decides to proceed, rewrite query, or switch strategy based on quality scores
- `route_after_evaluation`: Evaluates answer quality, decides to retry generation or return result

### 2. Iterative Self-Correction
**Traditional RAG**: Single-pass retrieval and generation<br>
**This System**: Two self-correction loops with quality gates

- **Retrieval Loop**: Poor quality (<0.6) -> 5 issue types + keyword injection -> rewrite query or switch strategy (max 2 attempts)
- **Generation Loop**: Consolidated evaluation (refusal + HHEM hallucination + quality) -> unified feedback -> regenerate with low temperature (max 3 attempts)
- **Early Strategy Switching**: off_topic/wrong_domain detected -> immediate strategy switch (saves 30-50% tokens)

### 3. Context Management
**Traditional RAG**: Stateless, no conversation memory<br>
**This System**: Persistent state across conversation turns

- Conversational rewrite transforms follow-up queries into self-contained questions
- MemorySaver checkpointer persists state across multi-turn conversations

### 4. Intelligent Tool Selection
**Traditional RAG**: Single retrieval method<br>
**This System**: Three retrieval strategies with intelligent selection

- **Strategies**: Semantic (FAISS), Keyword (BM25), Hybrid (RRF fusion)
- **Selection**: `decide_retrieval_strategy_node` analyzes corpus stats + query characteristics
- **Adaptation**: Switches strategies mid-execution based on content analysis

### Architecture Pattern

**No Central Agent Orchestrator**: The LangGraph StateGraph itself IS the agent. Decision-making is distributed across routing functions and conditional edges. This "Dynamic Planning and Execution Agents" pattern is more controllable and debuggable than single-agent orchestration while maintaining full autonomy through quality-driven routing.

### Research-Backed Enhancements

- **CRAG**: Confidence-based action triggering with early detection at retrieval stage
- **PreQRAG**: Strategy-specific query optimization (13-14% MRR improvement)
- **RAG-Fusion**: Multi-query retrieval with RRF ranking fusion (3-5% MRR improvement)
- **vRAG-Eval**: Answer quality evaluation with adaptive thresholds (65%/50% based on retrieval quality)
- **Hallucination Detection**: Claim decomposition + HHEM-2.1-Open verification (outperforms GPT-4)

## Architecture Overview

The system uses a 7-node LangGraph workflow with autonomous decision-making at every stage. Quality gates and routing functions provide distributed intelligence--each decision point evaluates intermediate results and autonomously determines the next action.

### Advanced RAG (7 nodes, 2 routers, 2 self-correction loops)

![Advanced RAG Architecture](mermaid%20chart.png)

**See [Interactive Demo](Advanced_Agentic_RAG.ipynb)** for routing logic deep-dive and live comparison runs.

### Node Summary

| Node | Purpose |
|------|---------|
| `conversational_rewrite` | Makes query self-contained using conversation history |
| `decide_strategy` | Selects optimal retrieval strategy (semantic/keyword/hybrid) |
| `query_expansion` | Generates query variations, optimizes for strategy |
| `retrieve_with_expansion` | RRF fusion + two-stage reranking + quality evaluation |
| `rewrite_and_refine` | Query enrichment via keyword injection for improved retrieval |
| `answer_generation` | Structured RAG prompting with quality-aware instructions |
| `evaluate_answer` | Consolidated refusal + HHEM hallucination + quality assessment |

## Features

The Advanced tier implements 17 features across retrieval, generation, and evaluation.

<details>
<summary><strong>Document & Corpus Profiling</strong></summary>

- LLM-based profiling of documents before chunking
- Analyzes technical density, document types, and domain characteristics
- Informs retrieval strategy selection
</details>

<details>
<summary><strong>Query Processing</strong></summary>

- **Conversational Rewriting**: Transforms follow-up queries into self-contained questions
- **Query Expansion**: Generates 3 variations (technical implementation, practical applications, conceptual principles)
- **Strategy-Specific Optimization**: Keyword -> specific terms; Semantic -> conceptual phrasing
</details>

<details>
<summary><strong>Intelligent Strategy Selection</strong></summary>

- Pure LLM-based classification (domain-agnostic, handles all edge cases)
- Analyzes query characteristics + corpus statistics
- Selects semantic/keyword/hybrid with confidence score + reasoning
</details>

<details>
<summary><strong>Multi-Strategy Retrieval</strong></summary>

- **Semantic**: FAISS vector search for meaning-based retrieval
- **Keyword**: BM25 lexical search for exact term matching
- **Hybrid**: Combines both with RRF-based fusion
- **RRF Multi-Query Fusion**: Aggregates rankings across query variants BEFORE reranking
</details>

<details>
<summary><strong>Two-Stage Reranking</strong></summary>

- **Stage 1**: CrossEncoder (ms-marco-MiniLM-L-6-v2) filters to top-10
- **Stage 2**: LLM-as-judge scores each document 0-100, selects top-4
- 3-5x faster than pure LLM reranking, 5-10x cheaper
</details>

<details>
<summary><strong>Quality Gates & Self-Correction</strong></summary>

- **Retrieval Quality**: 5 issue types (partial_coverage, missing_key_info, incomplete_context, wrong_domain, off_topic) + keyword injection
- **Answer Quality**: 5 issue types (incomplete_synthesis, lacks_specificity, missing_details, partial_answer, wrong_focus)
- **Adaptive Thresholds**: 65% for good retrieval, 50% for poor retrieval
</details>

<details>
<summary><strong>HHEM-Based Hallucination Detection</strong></summary>

- Claim decomposition: LLM extracts individual claims from answers
- HHEM verification: vectara/hallucination_evaluation_model (HHEM-2.1-Open) validates each claim
- Groundedness threshold: 0.5 (unsupported claims trigger regeneration)
</details>

<details>
<summary><strong>Multi-turn Conversations</strong></summary>

- Preserves conversation context with state persistence
- Thread management via MemorySaver checkpointer
- Automatic query contextualization
</details>

## Architecture Tiers

All tiers use the same **budget model tier** (GPT-4o-mini) to isolate architectural improvements from model quality.

| Tier | Features | Key Additions |
|------|----------|---------------|
| **Basic** | 1 | Semantic search only, direct LLM generation |
| **Intermediate** | 5 | + Query expansion, hybrid retrieval, CrossEncoder reranking, RRF fusion |
| **Advanced** | 17 | + Strategy selection, two-stage reranking, HHEM detection, quality gates, self-correction loops |
| **Multi-Agent** | 20 | + Query decomposition, parallel retrieval workers, cross-agent LLM relevance scoring |

**Run the comparison yourself:** See [Advanced_Agentic_RAG.ipynb](Advanced_Agentic_RAG.ipynb)

### When to Use Each Tier

- **Basic**: Simple factual lookups, low latency requirements
- **Intermediate**: Enhanced retrieval for predictable latency
- **Advanced**: Complex domains where query understanding matters
- **Multi-Agent**: Research synthesis, multi-faceted questions

## Quick Start

**Prerequisites:** Python 3.11+

```bash
# 1. Install dependencies (uses uv, not pip)
uv sync

# 2. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Run demo
uv run python main.py
```

## Model Tier Configuration

Control cost-quality tradeoffs via `MODEL_TIER` environment variable:

| Tier | Models | Use Case |
|------|--------|----------|
| **budget** | All GPT-4o-mini | Development, demos, architecture showcase |
| **balanced** | GPT-4o-mini + GPT-5-mini | Production (cost-conscious) |
| **premium** | GPT-5.1 + GPT-5-mini + GPT-5-nano | Production (quality-critical) |
 
```bash
# Set in .env
MODEL_TIER=budget    # Default - best cost-efficiency
MODEL_TIER=balanced  # Best cost-quality tradeoff
MODEL_TIER=premium   # Maximum quality
```

## Technology Stack

- **LLM Framework**: LangChain 1.0
- **Orchestration**: LangGraph 1.0 (StateGraph)
- **Vector Store**: FAISS
- **Lexical Search**: BM25
- **LLMs**: OpenAI GPT-4o-mini/GPT-5-mini/GPT-5.1/GPT-5-nano (configurable)
- **PDF Processing**: PyMuPDF
- **Reranking**: sentence-transformers (CrossEncoder)
- **Hallucination Detection**: HHEM-2.1-Open (vectara/hallucination_evaluation_model)
- **Package Manager**: uv

## Evaluation

### Metrics

- **Retrieval**: F1@K, Precision@K, Recall@K, MRR, nDCG
- **Generation**: Groundedness (HHEM-based), Semantic Similarity, Factual Accuracy, Completeness

### Golden Datasets

| Dataset | Questions | Avg Chunks | Cross-Doc | Query Types |
|---------|-----------|------------|-----------|-------------|
| **Standard** | 20 | 1.9 | 10% | factual, conceptual, procedural, comparative |
| **Hard** | 10 | 4.6 | 50% | procedural, comparative (multi-document) |

### Architecture Comparison Results

All tiers use **budget models** (GPT-4o-mini only) to isolate architectural improvements from model quality.

#### Standard Dataset (20 questions, k=4)

| Tier | Precision@4 | Recall@4 | F1@4 | MRR | nDCG@4 |
|------|-----|-----|------|-----|--------|
| Basic | 10.0% | 23.8% | 13.1% | 0.204 | 0.191 |
| Intermediate | 17.5% | 40.0% | 23.0% | 0.425 | 0.384 |
| Advanced | 20.0% | 43.3% | 25.9% | 0.550 | 0.443 |
| **Multi-Agent** | **23.8%** | **47.1%** | **29.6%** | **0.558** | **0.464** |

#### Hard Dataset (10 questions, k=6, multi-document)

| Tier | Precision@6 | Recall@6 | F1@6 | MRR | nDCG@6 |
|------|-----|-----|------|-----|--------|
| Basic | 25.0% | 35.6% | 29.0% | 0.553 | 0.365 |
| Intermediate | 23.3% | 33.1% | 27.0% | 0.533 | 0.358 |
| Advanced | 28.3% | 38.4% | 32.1% | 0.600 | 0.422 |
| **Multi-Agent** | **31.7%** | **42.3%** | **35.7%** | **0.667** | **0.464** |

## Future Improvements

- **HyDE**: Hypothetical document embeddings for better retrieval
- **Step-back prompting**: Higher-level conceptual questions for multi-hop reasoning
- **Chain-of-thought generation**: Structured reasoning with mandatory inline citations
- **Context compression**: Reduce prompt tokens by 75% while maintaining accuracy
- **LangSmith integration**: Production tracing, user feedback collection, quality dashboards
