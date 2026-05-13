from researchmind.tools.pdf_parser import (
    PDFParser
)

from researchmind.tools.chunker import (
    TextChunker
)

from researchmind.rag.embeddings import (
    EmbeddingGenerator
)

from researchmind.rag.vector_store import (
    VectorStore
)

from researchmind.utils.logger import (
    log_info
)


class RAGTool:
    def __init__(self):
        self.chunker = (
            TextChunker()
        )

        self.embedding_model = (
            EmbeddingGenerator()
        )

        self.vector_store = (
            VectorStore()
        )

    def ingest_pdf(
        self,
        pdf_path
    ):
        parser = PDFParser(
            pdf_path
        )

        text = (
            parser.extract_text()
        )

        chunks = (
            self.chunker
            .split_text(text)
        )

        embeddings = []

        for chunk in chunks:
            embedding = (
                self.embedding_model
                .generate_embedding(
                    chunk
                )
            )

            embeddings.append(
                embedding
            )

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings
        )

        log_info(
            f"PDF processed: "
            f"{pdf_path}"
        )

        return {
            "status": "success",
            "chunks": len(chunks)
        }