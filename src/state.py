from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AdvancedRAGState(TypedDict):
    """Enhanced state with query optimization"""
    
    # Input
    original_query: str
    
    # Query optimization
    query_expansions: Annotated[list[str], operator.add]  # Multiple query variants
    rewritten_query: str  # Rewritten for clarity
    current_query: str  # Query being used for retrieval
    
    # Retrieval strategy
    retrieval_strategy: str  # "semantic", "keyword", or "hybrid"
    
    # Processing
    messages: Annotated[list[BaseMessage], operator.add]
    retrieved_docs: Annotated[list[str], operator.add]
    retrieval_quality_score: float  # How good are the retrieved docs?
    
    # Decision making
    needs_retrieval: bool
    is_answer_sufficient: bool
    retrieval_attempts: int
    
    # Output
    final_answer: str
    confidence_score: float