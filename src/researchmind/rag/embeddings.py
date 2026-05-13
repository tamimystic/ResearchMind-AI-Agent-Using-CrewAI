import ollama


class EmbeddingGenerator:
    def __init__(
        self,
        model="nomic-embed-text"
    ):
        self.model = model

    def generate_embedding(
        self,
        text
    ):
        response = ollama.embeddings(
            model=self.model,
            prompt=text
        )

        return response[
            "embedding"
        ]