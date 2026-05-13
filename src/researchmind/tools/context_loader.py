from researchmind.rag.retriever import (
    Retriever
)


class ContextLoader:

    def __init__(self):
        self.retriever = (
            Retriever()
        )

    def load_context(
        self,
        query,
        top_k=4,
        max_chars=3500
    ):
        lower_query = (
            query.lower()
        )

        if any(
            keyword in lower_query
            for keyword in [
                "report",
                "implementation",
                "methodology"
            ]
        ):
            top_k = 5
            max_chars = 4500

        elif any(
            keyword in lower_query
            for keyword in [
                "summary",
                "math",
                "limitation"
            ]
        ):
            top_k = 3
            max_chars = 3000

        context = (
            self.retriever
            .retrieve(
                query=query,
                top_k=top_k
            )
        )

        context = context[
            :max_chars
        ]

        return f"""
Relevant Research Context:

{context}

Research Query:
{query}
""".strip()