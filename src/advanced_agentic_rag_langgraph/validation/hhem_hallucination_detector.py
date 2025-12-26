"""HHEM-2.1-Open hallucination detector: claim decomposition + per-chunk consistency verification."""

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import ctypes
import json
import re
import logging

# Suppress HuggingFace transformers warnings about custom model config (HHEMv2Config)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)


class ClaimDecomposition(TypedDict):
    """Structured output schema for claim extraction"""
    claims: List[str]
    reasoning: str


class HHEMHallucinationDetector:
    """HHEM-based hallucination detector. Score 0=hallucination, 1=consistent."""

    def __init__(
        self,
        hhem_model_name: str = "vectara/hallucination_evaluation_model",
        llm_model: str = None,
        entailment_threshold: float = 0.5
    ):
        """Initialize with HHEM model, LLM for claims, and support threshold (default 0.5)."""
        # HHEM-2.1 requires AutoModel, not CrossEncoder (shifted away from SentenceTransformers)
        self.hhem_model = AutoModelForSequenceClassification.from_pretrained(
            hhem_model_name,
            trust_remote_code=True
        )
        self.hhem_model.eval()  # Set to evaluation mode for inference

        # Tokenizer for input truncation (HHEM max sequence length is 512 tokens)
        # HHEM uses custom config that AutoTokenizer can't map, but it's based on FLAN-T5-Base
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        spec = get_model_for_task("hhem_claim_decomposition")
        llm_model = llm_model or spec.name

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=spec.temperature,
            max_tokens=1024,  # Safety net: prevent runaway generation
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity
        )
        self.structured_llm = self.llm.with_structured_output(ClaimDecomposition)
        self.entailment_threshold = entailment_threshold

    def _truncate_pair(self, claim: str, context: str) -> tuple:
        """Truncate context to fit within HHEM's 512 token limit."""
        MAX_TOTAL_TOKENS = 500

        claim_tokens = self.tokenizer.encode(claim, add_special_tokens=False)
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        max_context_tokens = MAX_TOTAL_TOKENS - len(claim_tokens)

        if len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
            context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)

        return (context, claim)

    def decompose_into_claims(self, answer: str) -> List[str]:
        """Decompose answer into atomic factual claims using LLM."""
        from advanced_agentic_rag_langgraph.prompts import get_prompt

        decomposition_prompt = get_prompt("hhem_claim_decomposition", answer=answer)

        try:
            result = self.structured_llm.invoke([HumanMessage(content=decomposition_prompt)])
            return result["claims"]
        except Exception as e:
            print(f"Warning: Claim decomposition failed: {e}. Using fallback.")
            # Fallback: Split by sentences and periods
            return [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]

    def verify_claim_entailment(self, claim: str, context: str) -> dict:
        """Verify single claim against context using HHEM. Auto-truncates to 512 token limit."""
        pair = self._truncate_pair(claim, context)

        with torch.inference_mode():
            scores = self.hhem_model.predict([pair])

        consistency_score = float(scores[0])
        supported = consistency_score >= self.entailment_threshold

        return {
            "entailment_score": consistency_score,
            "consistency_score": consistency_score,
            "label": "supported" if supported else "unsupported",
            "supported": supported
        }

    def verify_groundedness(self, answer: str, chunks: List[str]) -> dict:
        """Verify answer groundedness via per-chunk claim verification. Returns groundedness_score."""
        if not answer or not chunks:
            return {
                "claims": [],
                "entailment_scores": [],
                "supported": [],
                "unsupported_claims": [],
                "groundedness_score": 1.0,
                "claim_details": [],
                "reasoning": "Empty answer or chunks"
            }

        claims = self.decompose_into_claims(answer)

        if not claims:
            return {
                "claims": [],
                "entailment_scores": [],
                "supported": [],
                "unsupported_claims": [],
                "groundedness_score": 1.0,
                "claim_details": [],
                "reasoning": "No claims extracted from answer"
            }

        # Build all (context, claim) pairs for batch inference
        all_pairs = []
        for claim in claims:
            for chunk in chunks:
                truncated_pair = self._truncate_pair(claim, chunk)
                all_pairs.append(truncated_pair)

        # SINGLE batch inference call (was N claims x M chunks individual calls)
        with torch.inference_mode():
            all_scores = self.hhem_model.predict(all_pairs)

        # Reconstruct per-claim results from batch scores
        num_chunks = len(chunks)
        claim_details = []
        entailment_scores = []
        supported_flags = []
        unsupported_claims = []

        for claim_idx, claim in enumerate(claims):
            # Extract scores for this claim (consecutive in batch)
            start_idx = claim_idx * num_chunks
            chunk_scores = [float(all_scores[start_idx + i]) for i in range(num_chunks)]

            max_score = max(chunk_scores)
            best_chunk_idx = chunk_scores.index(max_score)
            supported = max_score >= self.entailment_threshold

            claim_details.append({
                "claim": claim,
                "entailment_score": max_score,
                "label": "supported" if supported else "unsupported",
                "supported": supported,
                "best_chunk_idx": best_chunk_idx,
                "chunk_scores": chunk_scores
            })

            entailment_scores.append(max_score)
            supported_flags.append(supported)

            if not supported:
                unsupported_claims.append(claim)

        total_claims = len(claims)
        supported_count = sum(supported_flags)
        groundedness_score = supported_count / total_claims if total_claims > 0 else 1.0
        reasoning = f"Verified {total_claims} claims against {len(chunks)} chunks: {supported_count} supported, {len(unsupported_claims)} unsupported"

        # Release memory to OS (glibc malloc_trim) - fixes Azure hang on 2nd evaluation
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except OSError:
            pass  # Windows/non-glibc systems

        return {
            "claims": claims,
            "entailment_scores": entailment_scores,
            "supported": supported_flags,
            "unsupported_claims": unsupported_claims,
            "groundedness_score": groundedness_score,
            "claim_details": claim_details,
            "reasoning": reasoning
        }
