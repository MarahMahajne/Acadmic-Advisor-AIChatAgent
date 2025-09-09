import logging
from PyPDF2 import PdfReader

class PDFProcessor:
    """
    PDF processing class to handle PDF file operations.
    Provides helper methods for validating file extensions and
    extracting textual content from PDF documents.
    """

    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        """
        Check if a given filename has an allowed extension.

        Args:
            filename (str): The name of the file to check.
            allowed_extensions (set): A set of allowed file extensions (e.g., {"pdf"}).

        Returns:
            bool: True if the file extension is allowed, False otherwise.
        """
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions
    

    @staticmethod
    def extract_text_from_pdf(file_stream) -> str:
        """
        Extract text from a PDF file stream.

        Args:
            file_stream: A file-like object pointing to the PDF (can be BytesIO or open file handle).

        Returns:
            str: The concatenated text from all pages in the PDF.
                 Each page is prefixed with a page marker like "--- Page X ---".

        Notes:
            - If extraction from a specific page fails, the method logs a warning
              and inserts a placeholder instead of crashing.
            - If the PDF itself cannot be opened, the exception is raised after logging an error.
        """
        try:
            # Initialize the PDF reader
            pdf_reader = PdfReader(file_stream)
            text_content = ""

            # Loop through each page in the PDF
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Try extracting text from the page
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    # If extraction fails for this page, log warning and insert placeholder
                    logging.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text_content += f"\n--- Page {page_num + 1} ---\n[Text extraction failed]\n"

            return text_content.strip()

        except Exception as e:
            # If the PDF cannot be opened at all
            logging.error(f"Failed to extract text from PDF: {e}")
            raise
