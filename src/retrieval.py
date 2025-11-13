from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import json
import re

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============ QUERY EXPANSION & REWRITING ============

def expand_query(query: str) -> list[str]:
    """Generate multiple query variations for better retrieval"""
    
    expansion_prompt = f"""For this question: "{query}"

Generate 3 alternative phrasings that would help retrieve better information:
- One more technical/formal version
- One simpler/layman's version  
- One focused on different aspect of the topic

Return as JSON:
{{
    "variations": [
        "variation 1",
        "variation 2", 
        "variation 3"
    ]
}}

Return ONLY the JSON, no explanation."""
    
    response = llm.invoke(expansion_prompt)
    
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {"variations": [query]}
        return [query] + data.get("variations", [])
    except:
        return [query]

def rewrite_query(query: str, retrieval_context: str = None) -> str:
    """Rewrite unclear queries for better retrieval accuracy"""
    
    context_info = ""
    if retrieval_context:
        context_info = f"\n\nPrevious retrieval returned insufficient results."
    
    rewrite_prompt = f"""Rewrite this query to be clearer and more specific for a search system:

Original: "{query}"{context_info}

Focus on:
- Adding missing context
- Using standard terminology
- Being concise but complete
- Removing ambiguity

Return ONLY the rewritten query."""
    
    response = llm.invoke(rewrite_prompt)
    return response.content.strip()

# ============ RERANKING ============

class ReRanker:
    """Rerank documents using LLM-as-Judge pattern"""
    
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def rank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """Rerank documents using relevance scoring"""
        
        if not documents:
            return []
        
        # Build ranking prompt
        doc_list = "\n".join([
            f"{i+1}. [{doc.metadata.get('source', 'unknown')}] {doc.page_content[:200]}..."
            for i, doc in enumerate(documents)
        ])
        
        ranking_prompt = f"""Query: "{query}"

Documents:
{doc_list}

For each document, rate its relevance to the query (0-100).
- 0-30: Not relevant
- 31-60: Somewhat relevant  
- 61-85: Relevant
- 86-100: Highly relevant

Return as JSON:
{{
    "scores": [score1, score2, score3, ...]
}}

Return ONLY the JSON."""
        
        response = self.llm.invoke(ranking_prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else {"scores": [50]*len(documents)}
            scores = data.get("scores", [50]*len(documents))
        except:
            scores = [50] * len(documents)
        
        # Pair documents with scores and sort
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:self.top_k]

# ============ HYBRID RETRIEVAL ============

class HybridRetriever:
    """Combine semantic + keyword retrieval"""
    
    def __init__(self, documents: list[Document]):
        self.semantic_retriever = SemanticRetriever(documents)
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        self.reranker = ReRanker(top_k=4)
    
    def retrieve(self, query: str, strategy: str = "hybrid") -> list[Document]:
        """
        Retrieve using specified strategy:
        - semantic: Dense vector search
        - keyword: BM25 lexical search
        - hybrid: Both combined
        """
        
        if strategy == "semantic":
            docs = self.semantic_retriever.retrieve(query, k=8)
        elif strategy == "keyword":
            docs = self.keyword_retriever.invoke(query)[:8]
        else:  # hybrid
            semantic_docs = self.semantic_retriever.retrieve(query, k=5)
            keyword_docs = self.keyword_retriever.invoke(query)[:5]
            
            # Deduplicate
            seen = set()
            docs = []
            for doc in semantic_docs + keyword_docs:
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in seen:
                    docs.append(doc)
                    seen.add(doc_id)
        
        # Rerank the results
        ranked_docs = self.reranker.rank(query, docs)
        return [doc for doc, score in ranked_docs]

class SemanticRetriever:
    """Semantic retrieval using embeddings"""
    
    def __init__(self, documents: list[Document]):        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(documents, embeddings)
    
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)