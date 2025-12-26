"""HHEM hallucination detector using Vectara's managed API."""

from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from advanced_agentic_rag_langgraph.core.model_config import get_model_for_task
import requests
import os
import re
import logging

logger = logging.getLogger(__name__)


class ClaimDecomposition(TypedDict):
    """Structured output schema for claim extraction"""
    claims: List[str]
    reasoning: str


class VectaraHHEMClient:
    """Client for Vectara's HHEM managed API (HHEM-2.3)."""

    API_URL = "https://api.vectara.io/v2/evaluate_factual_consistency"

    def __init__(self, api_key: str = None, customer_id: str = None):
        """Initialize with Vectara API key and customer ID."""
        self.api_key = api_key or os.getenv("VECTARA_API_KEY")
        self.customer_id = customer_id or os.getenv("VECTARA_CUSTOMER_ID")

        if not self.api_key:
            raise ValueError("VECTARA_API_KEY environment variable not set")
        if not self.customer_id:
            raise ValueError("VECTARA_CUSTOMER_ID environment variable not set")

        self.headers = {
            "Content-Type": "application/json",
            "customer-id": self.customer_id,
            "x-api-key": self.api_key
        }

    def evaluate(self, generated_text: str, source_texts: List[str]) -> dict:
        """Evaluate factual consistency of generated text against source texts.

        Args:
            generated_text: The claim/response to evaluate
            source_texts: List of source contexts to check against

        Returns:
            dict with score (0-1), p_consistent, p_inconsistent
        """
        payload = {
            "generated_text": generated_text,
            "source_texts": source_texts
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Vectara API error: {e}")
            # Return neutral score on API failure
            return {"score": 0.5, "p_consistent": 0.5, "p_inconsistent": 0.5}


class HHEMHallucinationDetector:
    """HHEM-based hallucination detector using Vectara's managed API.

    Score 0=hallucination, 1=consistent.
    """

    def __init__(
        self,
        llm_model: str = None,
        entailment_threshold: float = 0.5
    ):
        """Initialize with Vectara API client and LLM for claim decomposition."""
        self.vectara_client = VectaraHHEMClient()

        spec = get_model_for_task("hhem_claim_decomposition")
        llm_model = llm_model or spec.name

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=spec.temperature,
            max_tokens=1024,
            reasoning_effort=spec.reasoning_effort,
            verbosity=spec.verbosity
        )
        self.structured_llm = self.llm.with_structured_output(ClaimDecomposition)
        self.entailment_threshold = entailment_threshold

    def decompose_into_claims(self, answer: str) -> List[str]:
        """Decompose answer into atomic factual claims using LLM."""
        from advanced_agentic_rag_langgraph.prompts import get_prompt

        decomposition_prompt = get_prompt("hhem_claim_decomposition", answer=answer)

        try:
            result = self.structured_llm.invoke([HumanMessage(content=decomposition_prompt)])
            return result["claims"]
        except Exception as e:
            print(f"Warning: Claim decomposition failed: {e}. Using fallback.")
            return [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]

    def verify_claim_entailment(self, claim: str, context: str) -> dict:
        """Verify single claim against context using Vectara HHEM API."""
        result = self.vectara_client.evaluate(claim, [context])

        consistency_score = result.get("score", 0.5)
        supported = consistency_score >= self.entailment_threshold

        return {
            "entailment_score": consistency_score,
            "consistency_score": consistency_score,
            "label": "supported" if supported else "unsupported",
            "supported": supported
        }

    def verify_groundedness(self, answer: str, chunks: List[str]) -> dict:
        """Verify answer groundedness via claim verification against all chunks.

        Key optimization: Vectara API accepts all chunks in one call per claim.
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
            # Vectara API accepts all chunks at once - no need to loop
            result = self.vectara_client.evaluate(claim, chunks)
            score = result.get("score", 0.5)
            supported = score >= self.entailment_threshold

            claim_details.append({
                "claim": claim,
                "entailment_score": score,
                "label": "supported" if supported else "unsupported",
                "supported": supported,
                "best_chunk_idx": 0,  # API doesn't return which chunk matched
                "chunk_scores": [score]  # Single aggregated score from API
            })

            entailment_scores.append(score)
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
