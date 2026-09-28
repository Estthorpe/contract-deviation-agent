"""
Build data/sample_contract.pdf from data/sample_contract.txt using PyMuPDF.

The .txt file is the single source of truth for the sample contract; this
script derives the PDF from it so the two can never disagree.
"""

import textwrap
from pathlib import Path

import pymupdf

PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 in points
MARGIN = 56
FONT_SIZE = 10
LINE_HEIGHT = 14
WRAP_WIDTH = 95


def build_pdf(source, target):
    """
    Render a plain-text contract into a paginated A4 PDF.

    Args:
        source (Path): Path to the source .txt file.
        target (Path): Path where the .pdf will be written.

    Returns:
        int: The number of pages written.
    """
    text = source.read_text(encoding="utf-8")
    document = pymupdf.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN

    for paragraph in text.split("\n"):
        lines = textwrap.wrap(paragraph, WRAP_WIDTH) or [""]
        for line in lines:
            if y > PAGE_HEIGHT - MARGIN:
                page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = MARGIN
            page.insert_text((MARGIN, y), line, fontsize=FONT_SIZE, fontname="helv")
            y += LINE_HEIGHT

    document.save(target)
    page_count = document.page_count
    document.close()
    return page_count


def main():
    """Build the sample PDF and report the page count."""
    source = Path("data") / "sample_contract.txt"
    target = Path("data") / "sample_contract.pdf"
    pages = build_pdf(source, target)
    print(f"Wrote {target} ({pages} pages)")


if __name__ == "__main__":
    main()