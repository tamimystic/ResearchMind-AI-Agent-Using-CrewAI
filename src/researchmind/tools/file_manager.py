import os

from researchmind.utils.constants import (
    UPLOAD_FOLDER
)


class FileManager:
    def __init__(self):
        os.makedirs(
            UPLOAD_FOLDER,
            exist_ok=True
        )

    def get_pdf_files(self):
        files = []

        for file in os.listdir(
            UPLOAD_FOLDER
        ):
            if file.endswith(
                ".pdf"
            ):
                files.append(
                    os.path.join(
                        UPLOAD_FOLDER,
                        file
                    )
                )

        return files

    def latest_pdf(self):
        pdf_files = (
            self.get_pdf_files()
        )

        if not pdf_files:
            return None

        return max(
            pdf_files,
            key=os.path.getctime
        )