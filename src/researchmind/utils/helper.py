import os

from researchmind.utils.constants import (
    UPLOAD_FOLDER
)


def get_latest_pdf():
    pdf_files = []

    for file in os.listdir(
        UPLOAD_FOLDER
    ):
        if file.endswith(
            ".pdf"
        ):
            pdf_files.append(
                os.path.join(
                    UPLOAD_FOLDER,
                    file
                )
            )

    if not pdf_files:
        return None

    return max(
        pdf_files,
        key=os.path.getctime
    )