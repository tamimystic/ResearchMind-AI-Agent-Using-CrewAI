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
            Find methodology related sections.

            Focus on:
            dataset,
            preprocessing,
            feature extraction,
            architecture,
            algorithm,
            training process,
            evaluation metrics,
            workflow,
            implementation steps.
            """,

            "summary":
            """
            Find:
            abstract,
            introduction,
            research objective,
            contribution,
            key findings,
            conclusion.
            """,

            "math":
            """
            Find:
            equations,
            formulas,
            loss functions,
            optimization methods,
            mathematical concepts,
            feature extraction formulas.
            """,

            "limitation":
            """
            Find:
            limitations,
            weaknesses,
            future work,
            constraints,
            bias,
            scalability concerns.
            """,

            "implementation":
            """
            Find:
            implementation details,
            architecture,
            preprocessing,
            workflow,
            deployment,
            pipeline,
            training flow.
            """,

            "related":
            """
            Find sections discussing:

            related work,
            literature review,
            prior research,
            previous studies,
            comparison with previous models,
            baseline methods,
            state-of-the-art methods,
            existing approaches.

            Focus on:
            CNN,
            Vision Transformer (ViT),
            transfer learning,
            plant disease classification,
            image classification,
            feature extraction.

            Prefer:
            comparison tables,
            discussion of previous methods,
            survey discussions,
            benchmark comparisons.

            Avoid:
            conclusion,
            dataset details,
            implementation steps,
            training procedure,
            author biography,
            references,
            citation list.
            """,

            "report":
            """
            Find:
            abstract,
            methodology,
            results,
            findings,
            contribution,
            conclusion,
            limitations.
            """
        }

        lower_query = (
            query.lower()
        )

        for key in query_map:

            if key in lower_query:

                return f"""
                {query}

                {query_map[key]}
                """.strip()

        return query

    def retrieve(
        self,
        query,
        top_k=4
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
                top_k
            )
        )

        return "\n\n".join(
            results
        )