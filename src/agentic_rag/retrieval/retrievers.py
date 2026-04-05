from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .two_stage_reranker import TwoStageReRanker


class AdaptiveRetriever:
    """Adaptive multi-strategy retrieval with intelligent selection"""

    def __init__(self, documents: list[Document], k_final: int = 4):
        self.k_final = k_final
        self.semantic_retriever = SemanticRetriever(documents)
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        # TwoStageReRanker: CrossEncoder (top-10) then LLM-as-judge (k_final)
        self.reranker = TwoStageReRanker(k_cross_encoder=10, k_final=k_final)

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        k_semantic: int = 10,
        k_keyword: int = 10,
        k_total: int = 15
    ) -> list[Document]:
        """Retrieve using specified strategy (semantic/keyword/hybrid) with two-stage reranking."""

        if strategy == "semantic":
            docs = self.semantic_retriever.retrieve(query, k=k_total)
        elif strategy == "keyword":
            docs = self.keyword_retriever.invoke(query)[:k_total]
        else:  # hybrid
            semantic_docs = self.semantic_retriever.retrieve(query, k=k_semantic)
            keyword_docs = self.keyword_retriever.invoke(query)[:k_keyword]

            seen = set()
            docs = []
            for doc in semantic_docs + keyword_docs:
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in seen:
                    docs.append(doc)
                    seen.add(doc_id)

        ranked_docs = self.reranker.rank(query, docs)
        return [doc for doc, score in ranked_docs]

    def retrieve_without_reranking(
        self,
        query: str,
        strategy: str = "hybrid",
        k_semantic: int = 10,
        k_keyword: int = 10,
        k_total: int = 15
    ) -> list[Document]:
        """Retrieve WITHOUT reranking. Used for multi-query retrieval before RRF fusion."""

        if strategy == "semantic":
            docs = self.semantic_retriever.retrieve(query, k=k_total)
        elif strategy == "keyword":
            docs = self.keyword_retriever.invoke(query)[:k_total]
        else:  # hybrid
            semantic_docs = self.semantic_retriever.retrieve(query, k=k_semantic)
            keyword_docs = self.keyword_retriever.invoke(query)[:k_keyword]

            seen = set()
            docs = []
            for doc in semantic_docs + keyword_docs:
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in seen:
                    docs.append(doc)
                    seen.add(doc_id)

        return docs


class SemanticRetriever:
    """Semantic retrieval using embeddings"""

    def __init__(self, documents: list[Document]):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    def retrieve(self, query: str, k: int) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)
