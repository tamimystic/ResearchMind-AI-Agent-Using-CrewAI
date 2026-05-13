from researchmind.utils.constants import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class TextChunker:
    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(
        self,
        text
    ):
        chunks = []

        start = 0

        while start < len(text):
            end = (
                start +
                self.chunk_size
            )

            chunk = text[start:end]

            chunks.append(
                chunk
            )

            start += (
                self.chunk_size -
                self.overlap
            )

        return chunks