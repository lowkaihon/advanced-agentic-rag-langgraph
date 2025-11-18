"""
Answer Generation Prompts

Generates grounded answers using only retrieved context.

Research-backed optimizations:
- GPT-4o-mini (BASE): Explicit CoT scaffolding + step-by-step instructions
- GPT-5 (GPT5): Concise instructions, no CoT (saves 20-80% latency for 2.9% accuracy gain)
- Expected faithfulness: 72% (GPT-4o baseline) -> 80-85% (GPT-5, no CoT overhead)

Key principle: RAG systems should use ONLY retrieved context, no external knowledge.
"""

# System prompt template (shared, but instructions differ)
SYSTEM_PROMPT_TEMPLATE = """{hallucination_feedback}You are an AI assistant that answers questions based exclusively on retrieved documents. Your role is to provide accurate, well-grounded responses using only the information present in the provided context.

{quality_instruction}

Core Instructions:
1. Base your answer ONLY on the provided context - do not use external knowledge or make assumptions beyond what is explicitly stated
2. If the context does not contain sufficient information to answer the question, clearly state: "The provided context does not contain enough information to answer this question."
3. Provide direct, concise answers that extract and synthesize the relevant information
4. When helpful for clarity or verification, you may reference specific documents (e.g., "Document 2 explains that..." or "According to the retrieved information...")
5. Match your confidence level to the retrieval quality - acknowledge uncertainty when present"""


BASE_USER_MESSAGE = """<retrieved_context>
{formatted_context}
</retrieved_context>

<question>
{question}
</question>

<instructions>
Follow these steps to answer the question:

STEP 1: UNDERSTAND THE QUESTION
- Identify what information is being requested
- Note if it's a single-part or multi-part question
- Determine required level of detail

STEP 2: LOCATE RELEVANT INFORMATION
- Scan retrieved documents for relevant passages
- Identify which documents contain key information
- Note if information spans multiple documents

STEP 3: VERIFY SUFFICIENCY
- Check if context contains enough information to fully answer
- If insufficient, respond: "The provided context does not contain enough information to answer this question."
- Do NOT proceed if context is inadequate

STEP 4: SYNTHESIZE ANSWER
- Combine information from relevant passages
- Ensure all statements are grounded in retrieved context
- Use clear, direct language
- Reference specific documents when helpful for verification

STEP 5: QUALITY CHECK
- Verify answer addresses the question completely
- Confirm all claims are supported by retrieved documents
- Ensure no external knowledge was used
- Check that confidence matches retrieval quality
</instructions>

<answer>"""


GPT5_USER_MESSAGE = """<retrieved_context>
{formatted_context}
</retrieved_context>

<question>
{question}
</question>

<instructions>
1. Answer the question using ONLY information from the <retrieved_context> section above
2. If the context is insufficient, respond: "The provided context does not contain enough information to answer this question."
3. Provide a direct, accurate answer that synthesizes the relevant information
4. If multiple documents contain relevant information, combine insights appropriately
5. Do not make assumptions or inferences beyond what is explicitly stated
</instructions>

<answer>"""


def get_answer_generation_prompts(
    hallucination_feedback: str,
    quality_instruction: str,
    formatted_context: str,
    question: str,
    is_gpt5: bool = False
) -> tuple[str, str]:
    """
    Get system and user prompts for answer generation.

    Args:
        hallucination_feedback: Prepended feedback if retry_needed=True
        quality_instruction: Instructions based on retrieval quality
        formatted_context: Retrieved documents formatted as context
        question: User's query
        is_gpt5: If True, use GPT5 variant (no CoT scaffolding)

    Returns:
        Tuple of (system_prompt, user_message)
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        hallucination_feedback=hallucination_feedback,
        quality_instruction=quality_instruction
    )

    if is_gpt5:
        user_message = GPT5_USER_MESSAGE.format(
            formatted_context=formatted_context,
            question=question
        )
    else:
        user_message = BASE_USER_MESSAGE.format(
            formatted_context=formatted_context,
            question=question
        )

    return system_prompt, user_message
