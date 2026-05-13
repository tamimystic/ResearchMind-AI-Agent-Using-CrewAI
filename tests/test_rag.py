from researchmind.tools.rag_tool import (
    RAGTool
)

from researchmind.utils.helper import (
    get_latest_pdf
)


def test_rag_pipeline():
    pdf_path = (
        get_latest_pdf()
    )

    if not pdf_path:
        print(
            "No PDF found."
        )
        return

    rag_tool = (
        RAGTool()
    )

    result = (
        rag_tool
        .ingest_pdf(
            pdf_path
        )
    )

    print(result)


if __name__ == "__main__":
    test_rag_pipeline()