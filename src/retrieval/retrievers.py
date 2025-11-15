from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .two_stage_reranker import TwoStageReRanker


class HybridRetriever:
    """Combine semantic + keyword retrieval"""

    def __init__(self, documents: list[Document]):
        self.semantic_retriever = SemanticRetriever(documents)
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        # TwoStageReRanker: CrossEncoder (top-10) then LLM-as-judge (top-4)
        self.reranker = TwoStageReRanker(k_cross_encoder=10, k_final=4)

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        k_semantic: int = 5,
        k_keyword: int = 5,
        k_total: int = 8
    ) -> list[Document]:
        """
        Retrieve using specified strategy with configurable k-values.

        Args:
            query: Search query
            strategy: "semantic", "keyword", or "hybrid"
            k_semantic: Number of docs for semantic search (default: 5)
            k_keyword: Number of docs for keyword search (default: 5)
            k_total: Total docs for single-strategy retrieval (default: 8)

        Returns:
            List of reranked documents
        """

        if strategy == "semantic":
            docs = self.semantic_retriever.retrieve(query, k=k_total)
        elif strategy == "keyword":
            docs = self.keyword_retriever.invoke(query)[:k_total]
        else:  # hybrid
            semantic_docs = self.semantic_retriever.retrieve(query, k=k_semantic)
            keyword_docs = self.keyword_retriever.invoke(query)[:k_keyword]

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
