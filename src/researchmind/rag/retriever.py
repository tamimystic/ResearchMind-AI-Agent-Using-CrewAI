from researchmind.rag.embeddings import (
    EmbeddingGenerator
)

from researchmind.rag.vector_store import (
    VectorStore
)


class Retriever:
    def __init__(self):
        self.embedding_model = (
            EmbeddingGenerator()
        )

        self.vector_store = (
            VectorStore()
        )

    def retrieve(
        self,
        query,
        top_k=3
    ):
        query_embedding = (
            self.embedding_model
            .generate_embedding(
                query
            )
        )

        results = (
            self.vector_store
            .search(
                query_embedding,
                top_k
            )
        )

        return "\n\n".join(
            results
        )