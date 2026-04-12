import pymupdf as fitz


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF file.

    Returns list of (page_number, text) tuples. Page numbers are 1-based.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            text = "\n".join(
                line for line in text.splitlines() if line.strip()
            )
            if text.strip():
                pages.append((i + 1, text))
    return pages
