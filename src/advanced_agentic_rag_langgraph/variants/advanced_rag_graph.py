"""
Advanced RAG Graph (31 Features) - Full Agentic RAG Implementation.

This is the complete system with all advanced features enabled.
It re-exports the main graph from orchestration.graph for comparison testing.

Features (31 total):
- Hybrid retrieval (semantic + keyword)
- Strategy selection (semantic/keyword/hybrid)
- LLM-based expansion decision
- RRF multi-query fusion
- Two-stage reranking (CrossEncoder + LLM-as-judge)
- Conversational query rewriting
- Retrieval quality scoring (8 issue types)
- Answer quality framework (8 issue types)
- Query rewriting loop (issue-specific feedback, max 2 rewrites)
- NLI-based hallucination detection
- Three-tier groundedness routing (SEVERE/MODERATE/NONE)
- Root cause detection (LLM vs retrieval-caused hallucination)
- Hallucination correction loop (max 2 retries)
- Dual-tier strategy switching (early + late)
- Query optimization for new strategy
- Expansion regeneration on strategy change
- Content-driven issue → strategy mapping
- Adaptive thresholds (65% good retrieval, 50% poor)
- Document profiling metadata
- Complete metrics suite
- 4 specialized routing functions
- 9 nodes with conditional edges
- State management with selective accumulation
- Multi-turn conversation support
- Comprehensive logging and metrics
- Quality gates at every stage
- Self-correction loops
- Dynamic planning and execution
- Distributed intelligence
- Context-aware adaptation
- Research-backed patterns (CRAG, PreQRAG, vRAG-Eval)

Graph Structure: 9 nodes, 4 routing functions
- conversational_rewrite_node
- decide_strategy_node
- query_expansion_node
- retrieve_with_expansion_node
- rerank_node
- grade_documents_node
- answer_generation_node
- check_hallucination_node
- regenerate_grounded_answer_node

Routing Functions:
- route_after_query_expansion: Detects strategy changes
- route_after_retrieval: Early strategy switching, issue detection
- route_after_evaluation: Late strategy switching, quality gates
- route_after_groundedness_check: Three-tier hallucination routing

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph

__all__ = ["advanced_rag_graph"]
