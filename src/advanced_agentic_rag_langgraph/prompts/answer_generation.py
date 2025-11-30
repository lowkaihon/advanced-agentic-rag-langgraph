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
SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that answers questions based exclusively on retrieved documents. Your role is to provide accurate, well-grounded responses using only the information present in the provided context.

{quality_instruction}

Core Instructions:
1. Base your answer ONLY on the provided context - do not use external knowledge or make assumptions beyond what is explicitly stated
2. Answer what the context supports, then explicitly note what aspects are not covered (e.g., "The context explains X and Y. However, it does not provide information about Z.")
3. Only refuse entirely if the context contains NO relevant information at all
4. Provide direct, concise answers that extract and synthesize the relevant information
5. When helpful for clarity or verification, you may reference specific documents
6. Match your confidence level to the retrieval quality - acknowledge uncertainty when present"""


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

STEP 3: ASSESS CONTEXT COVERAGE
- Check which aspects of the question can be answered from context
- Identify any gaps or missing information
- Follow the quality-aware instructions in the system prompt based on retrieval confidence level

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
2. Follow the quality-aware instructions in the system prompt based on retrieval confidence level
3. Provide a direct, accurate answer that synthesizes the relevant information
4. If multiple documents contain relevant information, combine insights appropriately
5. Do not make assumptions or inferences beyond what is explicitly stated
</instructions>

<answer>"""


def get_answer_generation_prompts(
    quality_instruction: str,
    formatted_context: str,
    question: str,
    is_gpt5: bool = False,
    retry_feedback: str = "",
) -> tuple[str, str]:
    """
    Get system and user prompts for answer generation.

    LLM best practices: System prompt = behavioral guidance, User message = task content.
    - quality_instruction goes to system prompt (how to behave)
    - retry_feedback goes to user message (turn-specific content with previous answer)

    Args:
        quality_instruction: Behavioral instructions based on retrieval quality or retry mode
        formatted_context: Retrieved documents formatted as context
        question: User's query
        is_gpt5: If True, use GPT5 variant (no CoT scaffolding)
        retry_feedback: Content for retry (previous answer + issues) - goes in user message

    Returns:
        Tuple of (system_prompt, user_message)
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        quality_instruction=quality_instruction
    )

    # Build retry block for user message (if retry)
    retry_block = ""
    if retry_feedback:
        retry_block = f"""<retry_instructions>
{retry_feedback}
</retry_instructions>

"""

    # Select appropriate user message template
    if is_gpt5:
        user_message_content = GPT5_USER_MESSAGE.format(
            formatted_context=formatted_context,
            question=question
        )
    else:
        user_message_content = BASE_USER_MESSAGE.format(
            formatted_context=formatted_context,
            question=question
        )

    # Prepend retry block to user message
    user_message = retry_block + user_message_content

    return system_prompt, user_message
