from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .reranking import ReRanker


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
