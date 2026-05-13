from researchmind.rag.retriever import (
    Retriever
)


class ContextBuilder:
    def __init__(self):
        self.retriever = (
            Retriever()
        )

    def build_context(
        self,
        query,
        top_k=3
    ):
        context = (
            self.retriever
            .retrieve(
                query=query,
                top_k=top_k
            )
        )

        return f"""
Relevant Research Context:

{context}

Research Query:
{query}
""".strip()