"""
RAG Architecture Variants for A/B Testing.

This module contains four implementation tiers to showcase architectural value progression:
- pure_semantic_rag_graph: Simplest RAG (4 features, pure semantic search)
- basic_rag_graph: Simple RAG (8 features, linear flow)
- intermediate_rag_graph: Enhanced RAG (18 features, conditional routing)
- advanced_rag_graph: Full Agentic RAG (31 features, adaptive loops)

All variants use the same BUDGET model tier (gpt-4o-mini) to isolate
architectural improvements from model quality differences.
"""

from advanced_agentic_rag_langgraph.variants.pure_semantic_rag_graph import (
    pure_semantic_rag_graph,
)
from advanced_agentic_rag_langgraph.variants.basic_rag_graph import basic_rag_graph
from advanced_agentic_rag_langgraph.variants.intermediate_rag_graph import (
    intermediate_rag_graph,
)
from advanced_agentic_rag_langgraph.variants.advanced_rag_graph import advanced_rag_graph

__all__ = [
    "pure_semantic_rag_graph",
    "basic_rag_graph",
    "intermediate_rag_graph",
    "advanced_rag_graph",
]
