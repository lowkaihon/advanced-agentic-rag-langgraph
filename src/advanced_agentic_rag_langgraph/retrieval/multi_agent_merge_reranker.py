"""LLM-based document reranking for Multi-Agent RAG merge stage."""

from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task


class DocumentScore(TypedDict):
    """Individual document relevance score with explanation."""
    document_id: str           # Document identifier (e.g., "doc_0", "doc_1")
    relevance_score: float     # 0-100 relevance score
    reasoning: str             # Brief explanation for this document's score


class RankingResult(TypedDict):
    """Structured output for document reranking."""
    scored_documents: list[DocumentScore]  # Per-document scores with reasoning
    overall_reasoning: str  # Brief summary of ranking approach


class SetSelectionResult(TypedDict):
    """Structured output for set-wise document selection (complex queries)."""
    selected_document_ids: list[str]  # List of k document IDs (e.g., ["doc_0", "doc_2"])
    selection_reasoning: str          # Brief explanation of selection rationale


class MultiAgentMergeReRanker:
    """Score and rank documents from multi-worker retrieval results.

    Supports query-adaptive merge strategy:
    - Simple queries: Individual pointwise scoring (precision-optimized)
    - Complex queries: Set-wise coverage selection (coverage-optimized)
    """

    def __init__(self, top_k: int = 6):
        """Initialize with top_k documents to return."""
        self.top_k = top_k

        spec = get_model_for_task("multi_agent_merge_reranking")

        self.llm = ChatOpenAI(
            model=spec.name,
            temperature=spec.temperature,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity,
        )
        self.structured_llm = self.llm.with_structured_output(RankingResult)
        self.selection_llm = self.llm.with_structured_output(SetSelectionResult)

    def rerank(
        self,
        original_question: str,
        candidate_docs: list[Document],
        sub_queries: list[str],
    ) -> tuple[list[Document], list[float] | None]:
        """Select documents using set-wise coverage selection.

        Multi-agent merge only runs for complex queries (simple queries use
        single-worker fast path), so we always use coverage-aware selection.

        Args:
            original_question: The original user question
            candidate_docs: Documents to select from
            sub_queries: Sub-queries/aspects for coverage selection

        Returns:
            (top-k docs, None) - no scores for set selection
        """
        if not candidate_docs:
            return [], None

        if len(candidate_docs) <= self.top_k:
            return candidate_docs, None  # No selection needed

        return self._set_selection(original_question, candidate_docs, sub_queries)

    def _set_selection(
        self,
        original_question: str,
        candidate_docs: list[Document],
        sub_queries: list[str],
    ) -> tuple[list[Document], list[float] | None]:
        """Select document SET for complex queries (coverage-aware).

        Uses set-wise selection to ensure coverage across all facets of
        comparative/multi-hop queries.
        """
        from advanced_agentic_rag_langgraph.prompts import get_prompt

        # Build document ID mapping
        doc_id_map = {f"doc_{i}": doc for i, doc in enumerate(candidate_docs)}

        # Format documents for prompt
        doc_list = []
        for i, doc in enumerate(candidate_docs):
            doc_id = f"doc_{i}"
            source = doc.metadata.get("source", "unknown")
            content_preview = doc.page_content[:1000]
            doc_list.append(f"{doc_id}: [Source: {source}]\n{content_preview}")

        # Format sub-queries as aspects list
        sub_queries_list = "\n".join([f"- {sq}" for sq in sub_queries])

        prompt = get_prompt(
            "multi_agent_merge_reranking_coverage",
            k=self.top_k,
            original_question=original_question,
            sub_queries_list=sub_queries_list,
            doc_list="\n\n".join(doc_list),
        )

        # Log input candidates
        print(f"\n{'='*60}")
        print(f"SET-WISE COVERAGE SELECTION (Multi-Agent Merge)")
        print(f"Original question: {original_question}")
        print(f"Aspects: {len(sub_queries)}")
        for sq in sub_queries:
            print(f"  - {sq}")
        print(f"Candidates: {len(candidate_docs)}")
        print(f"\nChunk IDs before selection:")
        for i, doc in enumerate(candidate_docs):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id}")

        try:
            result = self.selection_llm.invoke([HumanMessage(content=prompt)])
            selected_ids = result["selected_document_ids"]
            reasoning = result["selection_reasoning"]

            print(f"\nSelection reasoning: {reasoning}")
            print(f"\nSelected document IDs: {selected_ids}")

            # Map selected IDs to documents (preserve selection order)
            selected_docs = []
            for doc_id in selected_ids:
                if doc_id in doc_id_map:
                    selected_docs.append(doc_id_map[doc_id])
                else:
                    print(f"Warning: Invalid document_id '{doc_id}' in selection")

            # If not enough valid selections, pad with remaining docs
            if len(selected_docs) < self.top_k:
                print(f"Warning: Only {len(selected_docs)} valid selections, padding to {self.top_k}")
                remaining = [doc for doc in candidate_docs if doc not in selected_docs]
                selected_docs.extend(remaining[:self.top_k - len(selected_docs)])

            # Log final selection
            print(f"\nFinal selection (top-{self.top_k}):")
            for i, doc in enumerate(selected_docs[:self.top_k]):
                chunk_id = doc.metadata.get("id", "unknown")
                print(f"  {i+1}. {chunk_id}")
            print(f"{'='*60}\n")

            return selected_docs[:self.top_k], None  # No scores for set selection

        except Exception as e:
            print(f"Warning: Set selection failed: {e}. Falling back to pointwise scoring.")
            return self._pointwise_scoring(original_question, candidate_docs, None)

    def _pointwise_scoring(
        self,
        original_question: str,
        candidate_docs: list[Document],
        fallback_scores: list[float] = None,
    ) -> tuple[list[Document], list[float] | None]:
        """Score documents individually (simple queries, precision-optimized)."""
        from advanced_agentic_rag_langgraph.prompts import get_prompt

        # Build document ID mapping (index-based for score lookup)
        doc_id_map = {f"doc_{i}": i for i, doc in enumerate(candidate_docs)}

        # Format documents for prompt
        doc_list = []
        for i, doc in enumerate(candidate_docs):
            doc_id = f"doc_{i}"
            source = doc.metadata.get("source", "unknown")

            content_preview = doc.page_content[:1000]
            doc_list.append(f"{doc_id}: [Source: {source}]\n{content_preview}")

        # Pass document count info for completeness enforcement
        doc_count = len(candidate_docs)
        expected_ids = ", ".join([f"doc_{i}" for i in range(doc_count)])
        last_doc_idx = doc_count - 1

        prompt = get_prompt(
            "multi_agent_merge_reranking",
            original_question=original_question,
            doc_list="\n\n".join(doc_list),
            doc_count=doc_count,
            expected_ids=expected_ids,
            last_doc_idx=last_doc_idx,
        )

        # Log input candidates
        print(f"\n{'='*60}")
        print(f"POINTWISE SCORING (Multi-Agent Merge)")
        print(f"Original question: {original_question}")
        print(f"Candidates: {len(candidate_docs)}")
        print(f"\nChunk IDs before scoring:")
        for i, doc in enumerate(candidate_docs):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id}")

        # Prepare fallback scores (position-based if not provided)
        if fallback_scores is None:
            fallback_scores = [100 - (i * 5) for i in range(len(candidate_docs))]

        used_llm_scores = False  # Track if we got valid LLM scores

        try:
            result = self.structured_llm.invoke([HumanMessage(content=prompt)])
            scored_docs = result["scored_documents"]

            # Create score mapping using document_id strings
            score_map = {}
            for doc_score in scored_docs:
                doc_id = doc_score["document_id"]
                if doc_id in doc_id_map:
                    idx = doc_id_map[doc_id]
                    score_map[idx] = doc_score["relevance_score"]
                else:
                    print(f"Warning: Invalid document_id '{doc_id}' in LLM response (expected: {list(doc_id_map.keys())})")

            # If incomplete scoring, fall back to position-based scores
            if len(score_map) != len(candidate_docs):
                missing = [f"doc_{i}" for i in range(len(candidate_docs)) if i not in score_map]
                print(f"Warning: Incomplete LLM scoring ({len(score_map)}/{len(candidate_docs)} docs scored, missing: {missing})")
                print(f"Falling back to position-based ranking")
                scores = fallback_scores
            else:
                # Complete scoring - use LLM scores
                scores = [score_map[i] for i in range(len(candidate_docs))]
                used_llm_scores = True

        except Exception as e:
            print(f"Warning: Reranking failed: {e}. Using fallback scores.")
            scores = fallback_scores

        # Log all scores
        print(f"\nLLM Scores (all {len(candidate_docs)} candidates):")
        for i, (doc, score) in enumerate(zip(candidate_docs, scores)):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id} (score: {score:.1f})")

        # Sort by score and return top_k
        ranked = sorted(
            zip(candidate_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log final selection
        top_k = ranked[:self.top_k]
        print(f"\nFinal selection (top-{self.top_k}):")
        for i, (doc, score) in enumerate(top_k):
            chunk_id = doc.metadata.get("id", "unknown")
            print(f"  {i+1}. {chunk_id} (score: {score:.1f})")
        print(f"{'='*60}\n")

        top_k_docs = [doc for doc, _ in top_k]
        top_k_scores = [score for _, score in top_k] if used_llm_scores else None
        return top_k_docs, top_k_scores
