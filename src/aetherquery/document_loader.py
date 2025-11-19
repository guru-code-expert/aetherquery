import pdfplumber
from typing import List

def load_pdf_to_chunks(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Extracts text from a PDF and splits it into overlapping chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]

    chunks: List[str] = []
    for page in pages_text:
        start = 0
        while start < len(page):
            end = start + chunk_size
            chunks.append(page[start:end])
            start = end - chunk_overlap
    return chunks