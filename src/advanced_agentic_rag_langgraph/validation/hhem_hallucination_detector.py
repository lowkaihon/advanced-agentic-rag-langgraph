"""
Hallucination detection for RAG systems using HHEM-2.1-Open.

Uses Vectara's HHEM-2.1-Open (Hallucination Evaluation Model) to verify
factual consistency of generated answers against retrieved context.

Model: vectara/hallucination_evaluation_model (HHEM-2.1-Open)
- #1 hallucination detection model on HuggingFace
- Trained specifically for RAG factual consistency (not general NLI)
- Outperforms GPT-3.5-Turbo and GPT-4 for hallucination detection
- Outputs: Single consistency score (0=hallucination, 1=consistent)
- Handles paraphrases correctly (unlike zero-shot NLI models)

Architecture:
1. Claim Decomposition: LLM extracts atomic factual claims from answer
2. HHEM Verification: Verifies each claim against context
3. Groundedness Score: Fraction of claims supported by context

Why HHEM over Zero-Shot NLI:
Zero-shot NLI (nli-deberta-v3-base) achieves only 0.65-0.70 F1 on RAG
content due to paraphrase handling issues. Claims like "Adam optimizer
was used" get flagged as NEUTRAL vs "We used the Adam optimizer" despite
identical meaning. HHEM is specifically trained for semantic equivalence
in RAG contexts, eliminating these false positives.
"""

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
from transformers import AutoModelForSequenceClassification
import torch
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
    """
    HHEM-based hallucination detector using claim decomposition + consistency verification.

    Uses Vectara's HHEM-2.1-Open, the #1 hallucination detection model on HuggingFace.
    HHEM is specifically trained for RAG factual consistency and handles paraphrases
    correctly, unlike zero-shot NLI models which produce high false positive rates.

    Output: Single consistency score (0=hallucination, 1=consistent)
    Threshold: Claims with score >= threshold are considered SUPPORTED
    """

    def __init__(
        self,
        hhem_model_name: str = "vectara/hallucination_evaluation_model",
        llm_model: str = None,
        entailment_threshold: float = 0.5
    ):
        """
        Initialize HHEM-based hallucination detector with tier-based model configuration.

        Uses HHEM-2.1-Open which outperforms GPT-3.5/GPT-4 for hallucination detection.

        Args:
            hhem_model_name: HHEM model for consistency verification (default: HHEM-2.1-Open)
            llm_model: LLM for claim decomposition (None = use tier config)
            entailment_threshold: Minimum consistency score to consider claim supported (0.5 = balanced)
        """
        # HHEM-2.1 requires AutoModel, not CrossEncoder (shifted away from SentenceTransformers)
        self.hhem_model = AutoModelForSequenceClassification.from_pretrained(
            hhem_model_name,
            trust_remote_code=True
        )
        self.hhem_model.eval()  # Set to evaluation mode for inference

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

    def decompose_into_claims(self, answer: str) -> List[str]:
        """
        Decompose answer into atomic factual claims using LLM.

        Each claim should be atomic (single fact), verifiable, and self-contained.
        """
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
        """
        Verify if claim is supported by context using HHEM consistency model.

        HHEM outputs a single consistency score (0=hallucination, 1=consistent),
        not 3-class NLI labels. This is specifically designed for RAG factual
        consistency and handles paraphrases correctly.

        Returns dict with consistency_score, label, and supported flag.
        """
        # HHEM format: List[Tuple[premise, hypothesis]] = List[Tuple[context, claim]]
        pairs = [(context, claim)]

        with torch.no_grad():
            scores = self.hhem_model.predict(pairs)

        # HHEM outputs single consistency score (0-1)
        # Higher = more consistent with context
        consistency_score = float(scores[0])
        supported = consistency_score >= self.entailment_threshold

        return {
            "entailment_score": consistency_score,  # Keep key for compatibility
            "consistency_score": consistency_score,
            "label": "supported" if supported else "unsupported",
            "supported": supported
        }

    def verify_groundedness(self, answer: str, chunks: List[str]) -> dict:
        """
        Verify groundedness of answer using per-chunk HHEM verification.

        Per-chunk verification ensures each HHEM call stays under 512 tokens.
        For each claim, verifies against all chunks and takes max score.

        Args:
            answer: The generated answer to verify
            chunks: List of individual chunk texts (not concatenated)

        Returns dict with claims, entailment_scores, supported flags, unsupported_claims,
        groundedness_score (supported/total), claim_details, and reasoning.
        """
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

        claim_details = []
        entailment_scores = []
        supported_flags = []
        unsupported_claims = []

        for claim in claims:
            # Per-chunk verification: check claim against each chunk, take max score
            chunk_scores = []
            best_chunk_idx = 0
            best_score = 0.0

            for idx, chunk in enumerate(chunks):
                verification = self.verify_claim_entailment(claim, chunk)
                score = verification["consistency_score"]
                chunk_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_chunk_idx = idx

            max_score = max(chunk_scores) if chunk_scores else 0.0
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

        return {
            "claims": claims,
            "entailment_scores": entailment_scores,
            "supported": supported_flags,
            "unsupported_claims": unsupported_claims,
            "groundedness_score": groundedness_score,
            "claim_details": claim_details,
            "reasoning": reasoning
        }
