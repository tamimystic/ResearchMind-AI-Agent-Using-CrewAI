import chromadb

from researchmind.utils.constants import (
    VECTOR_DB_FOLDER
)


class VectorStore:
    def __init__(
        self,
        collection_name="researchmind"
    ):
        self.client = (
            chromadb.PersistentClient(
                path=VECTOR_DB_FOLDER
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def add_chunks(
        self,
        chunks,
        embeddings
    ):
        ids = [
            f"chunk_{i}"
            for i in range(
                len(chunks)
            )
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

    def search(
        self,
        query_embedding,
        top_k=3
    ):
        results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k
            )
        )

        return results[
            "documents"
        ][0]