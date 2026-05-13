from researchmind.tools.rag_tool import (
    RAGTool
)

from researchmind.utils.helper import (
    get_latest_pdf
)

from researchmind.utils.logger import (
    log_info
)


class PaperIngestor:
    def __init__(self):
        self.rag_tool = (
            RAGTool()
        )

    def ingest_latest_paper(
        self
    ):
        pdf_path = (
            get_latest_pdf()
        )

        if not pdf_path:
            raise FileNotFoundError(
                "No PDF found in "
                "uploaded_papers"
            )

        result = (
            self.rag_tool
            .ingest_pdf(
                pdf_path
            )
        )

        log_info(
            f"Paper ingested: "
            f"{pdf_path}"
        )

        return result