"""
Coverage-aware document selection for Multi-Agent RAG merge stage.

Selects documents for maximum coverage across sub-query aspects,
unlike standard reranking which scores individual documents.

Key differences from LLMMetadataReRanker:
- Input: Original question + sub-queries (not single query)
- Goal: Select SET for coverage (not rank individuals)
- Output: Selected document IDs (not scored list)
- Criteria: Coverage + non-redundancy (not type/domain matching)
"""

from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


class DocCoverageReason(TypedDict):
    """Per-document coverage explanation."""
    document_id: str
    addresses_aspect: str  # Which sub-query/aspect this doc covers


class CoverageSelectionResult(TypedDict):
    """Structured output for coverage-aware document selection."""
    selected_document_ids: list[str]  # Exactly k document IDs
    coverage_analysis: str  # How selected docs cover the question
    per_doc_reasoning: list[DocCoverageReason]


class MultiAgentMergeReRanker:
    """Select documents for coverage across sub-query aspects."""

    def __init__(self, top_k: int = 6):
        """
        Initialize coverage-aware merge reranker.

        Args:
            top_k: Number of documents to select for final answer generation
        """
        self.top_k = top_k

        spec = get_model_for_task("multi_agent_merge_reranking")

        self.llm = ChatOpenAI(
            model=spec.name,
            temperature=spec.temperature,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity,
        )
        self.structured_llm = self.llm.with_structured_output(CoverageSelectionResult)

    def select_for_coverage(
        self,
        original_question: str,
        sub_queries: list[str],
        candidate_docs: list[Document],
        fallback_doc_ids: list[str] = None,
    ) -> list[Document]:
        """
        Select top-k documents for maximum coverage of the original question.

        Args:
            original_question: The user's original complex question
            sub_queries: List of decomposed sub-queries
            candidate_docs: Pre-filtered candidates (e.g., top-12 from RRF)
            fallback_doc_ids: Fallback selection if LLM fails (document IDs in order)

        Returns:
            List of selected Documents (exactly top_k)
        """
        if len(candidate_docs) <= self.top_k:
            return candidate_docs

        from advanced_agentic_rag_langgraph.prompts import get_prompt

        # Build document ID mapping
        doc_id_map = {f"doc_{i}": doc for i, doc in enumerate(candidate_docs)}

        # Format documents for prompt
        doc_list = []
        for i, doc in enumerate(candidate_docs):
            doc_id = f"doc_{i}"
            source = doc.metadata.get("source", "unknown")
            # Extract just filename from path
            if "/" in source:
                source = source.split("/")[-1]
            elif "\\" in source:
                source = source.split("\\")[-1]

            content_preview = doc.page_content[:800]
            doc_list.append(f"{doc_id}: [Source: {source}]\n{content_preview}")

        # Format sub-queries
        sub_queries_list = "\n".join([f"  {i+1}. {sq}" for i, sq in enumerate(sub_queries)])

        prompt = get_prompt(
            "multi_agent_merge_reranking",
            original_question=original_question,
            sub_queries_list=sub_queries_list,
            doc_list="\n\n".join(doc_list),
            k=self.top_k,
        )

        try:
            result = self.structured_llm.invoke([HumanMessage(content=prompt)])
            selected_ids = result["selected_document_ids"]

            # Validate and collect selected docs
            selected_docs = []
            for doc_id in selected_ids:
                if doc_id in doc_id_map:
                    selected_docs.append(doc_id_map[doc_id])
                else:
                    print(f"Warning: Invalid document_id '{doc_id}' in LLM response")

            # If incomplete, pad with fallback (RRF order)
            if len(selected_docs) < self.top_k:
                missing_count = self.top_k - len(selected_docs)
                print(f"Warning: LLM selected {len(selected_docs)}/{self.top_k} docs, padding with RRF fallback")

                # Add docs from fallback that aren't already selected
                fallback_ids = fallback_doc_ids or [f"doc_{i}" for i in range(len(candidate_docs))]
                for doc_id in fallback_ids:
                    if doc_id in doc_id_map and doc_id_map[doc_id] not in selected_docs:
                        selected_docs.append(doc_id_map[doc_id])
                        if len(selected_docs) >= self.top_k:
                            break

            return selected_docs[:self.top_k]

        except Exception as e:
            print(f"Warning: Coverage selection failed: {e}. Using RRF fallback.")
            # Fallback to RRF order (first top_k candidates)
            return candidate_docs[:self.top_k]
