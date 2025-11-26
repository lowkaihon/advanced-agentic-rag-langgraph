"""
Advanced RAG Graph (17 Features) - Full Agentic RAG Implementation.

This is the complete system with all advanced features enabled.
It re-exports the main graph from orchestration.graph for comparison testing.

Features (17 = +12 over Intermediate):

Inherited from Intermediate (5):
1. Semantic vector search
2. Query expansion (multi-variant)
3. Hybrid retrieval (semantic + BM25)
4. RRF fusion
5. CrossEncoder reranking

Intelligent Query Processing (+2):
6. Conversational query rewriting
7. LLM-based strategy selection

Enhanced Retrieval (+1):
8. Two-stage reranking (CrossEncoder -> LLM-as-judge)

Quality-Driven Routing (+3):
9. Retrieval quality gates (8 issue types)
10. Answer quality evaluation (8 issue types)
11. Adaptive thresholds (65%/50%)

Self-Correction Loops (+3):
12. Query rewriting loop (issue-specific feedback, max 3)
13. Early strategy switching (off_topic/wrong_domain)
14. Generation retry loop (adaptive temperature)

Anti-Hallucination (+2):
15. NLI-based hallucination detection
16. Refusal detection

Multi-Turn (+1):
17. Conversation context preservation

Graph Structure: 7 nodes, 2 routing functions
- conversational_rewrite_node
- decide_strategy_node
- query_expansion_node
- retrieve_with_expansion_node
- rewrite_and_refine_node
- answer_generation_node
- evaluate_answer_node

Routing Functions:
- route_after_retrieval: Early strategy switching, issue detection
- route_after_evaluation: Generation retry or end

All features use BUDGET model tier (gpt-4o-mini) for fair comparison.
"""

from advanced_agentic_rag_langgraph.orchestration.graph import advanced_rag_graph

__all__ = ["advanced_rag_graph"]
