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

    def _enhance_query(
        self,
        query
    ):
        query_map = {
            "methodology":
            """
            Focus on methodology section.

            Include:
            dataset,
            data collection,
            preprocessing,
            augmentation,
            feature extraction,
            architecture,
            workflow,
            algorithm,
            training process,
            optimizer,
            hyperparameters,
            implementation,
            evaluation metrics,
            experiment setup,
            classification pipeline.
            """,

            "summary":
            """
            Focus on:
            abstract,
            introduction,
            research problem,
            objective,
            contribution,
            motivation,
            key findings,
            conclusion.
            """,

            "math":
            """
            Focus on:
            equations,
            formulas,
            mathematical model,
            loss function,
            optimization,
            feature extraction equations,
            transformer equations,
            accuracy calculation.
            """,

            "limitation":
            """
            Focus on:
            limitations,
            weakness,
            future work,
            constraints,
            scalability,
            challenges,
            bias,
            drawbacks.
            """,

            "implementation":
            """
            Focus on:
            implementation details,
            architecture,
            pipeline,
            workflow,
            preprocessing,
            model setup,
            deployment,
            system design.
            """,

            "dataset":
            """
            Focus on:
            dataset,
            data acquisition,
            preprocessing,
            augmentation,
            train test split,
            data distribution,
            class information.
            """
        }

        lower_query = (
            query.lower()
        )

        for key, value in query_map.items():
            if key in lower_query:
                return f"""
                {query}

                {value}
                """.strip()

        return query

    def _rerank_results(
        self,
        results,
        query
    ):
        keyword_priority = [
            "methodology",
            "dataset",
            "data acquisition",
            "preprocessing",
            "augmentation",
            "feature extraction",
            "transfer learning",
            "vision transformer",
            "vit",
            "architecture",
            "workflow",
            "training",
            "optimizer",
            "epoch",
            "evaluation",
            "classification",
            "accuracy",
            "loss"
        ]

        negative_keywords = [
            "results",
            "discussion",
            "comparison",
            "table",
            "accuracy improved",
            "performance comparison"
        ]

        query_lower = (
            query.lower()
        )

        scored_results = []

        for chunk in results:
            score = 0
            chunk_lower = (
                chunk.lower()
            )

            for keyword in keyword_priority:
                if keyword in chunk_lower:
                    score += 1

            for word in query_lower.split():
                if word in chunk_lower:
                    score += 1

            for keyword in negative_keywords:
                if keyword in chunk_lower:
                    score -= 2

            scored_results.append(
                (
                    score,
                    chunk
                )
            )

        scored_results.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        return [
            chunk
            for _, chunk
            in scored_results
        ]

    def retrieve(
        self,
        query,
        top_k=3
    ):
        enhanced_query = (
            self._enhance_query(
                query
            )
        )

        query_embedding = (
            self.embedding_model
            .generate_embedding(
                enhanced_query
            )
        )

        results = (
            self.vector_store
            .search(
                query_embedding,
                top_k * 2
            )
        )

        reranked_results = (
            self._rerank_results(
                results,
                query
            )
        )

        return "\n\n".join(
            reranked_results[:top_k]
        )