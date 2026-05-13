import fitz


class PDFParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        text = ""

        document = fitz.open(
            self.file_path
        )

        for page in document:
            text += (
                page.get_text() + "\n"
            )

        document.close()

        return text.strip()