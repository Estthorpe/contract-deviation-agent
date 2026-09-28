"""
PDF parser tool for the Contract Deviation Detection Agent.

Turns whatever the user supplies (a PDF file, a text file, raw bytes from a
Streamlit upload, or a pasted string) into a single plain-text string that the
ClauseExtractorAgent can read. PDF text extraction uses PyMuPDF (imported as
``pymupdf`` — the ``fitz`` alias is deprecated in current releases).
"""

from pathlib import Path

import pymupdf

# Below this many characters we assume the PDF is an image scan with no text layer.
MIN_EXTRACTED_CHARS = 100


def _text_from_pdf_document(document):
    """
    Join the text of every page in an open PyMuPDF document.

    Args:
        document (pymupdf.Document): An open PDF document.

    Returns:
        str: Page texts joined with a blank line between pages.
    """
    pages = [page.get_text() for page in document]
    return "\n\n".join(pages)


def _extract_from_pdf_path(path):
    """
    Extract text from a PDF file on disk.

    Args:
        path (Path): Path to a .pdf file.

    Returns:
        str: The extracted text.

    Raises:
        ValueError: If PyMuPDF cannot open or read the file.
    """
    try:
        with pymupdf.open(path) as document:
            return _text_from_pdf_document(document)
    except Exception as error:
        raise ValueError(
            f"PyMuPDF could not read the PDF at '{path}'. Check that the file "
            f"exists, is a genuine PDF and is not password-protected. "
            f"Underlying error: {error}"
        ) from error


def _extract_from_pdf_bytes(data, label):
    """
    Extract text from PDF bytes held in memory (e.g. a Streamlit upload).

    Args:
        data (bytes): The raw PDF content.
        label (str): A name for the source, used in error messages.

    Returns:
        str: The extracted text.

    Raises:
        ValueError: If PyMuPDF cannot open or read the bytes as a PDF.
    """
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            return _text_from_pdf_document(document)
    except Exception as error:
        raise ValueError(
            f"PyMuPDF could not read the uploaded PDF '{label}'. Check that the "
            f"upload is a genuine PDF and is not password-protected. "
            f"Underlying error: {error}"
        ) from error


def _check_not_scanned(text, label):
    """
    Raise a clear error if a PDF yielded almost no text (an image scan).

    Args:
        text (str): The extracted text.
        label (str): A name for the source, used in the error message.

    Returns:
        str: The same text, unchanged, if it passes the check.

    Raises:
        ValueError: If fewer than MIN_EXTRACTED_CHARS non-blank characters were found.
    """
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"'{label}' appears to be an image scan - text extraction found fewer "
            f"than {MIN_EXTRACTED_CHARS} characters. Text extraction is not "
            f"supported for scanned PDFs. Please provide a text-based PDF or "
            f"paste the contract text directly."
        )
    return text


def extract_text(source, filename=None):
    """
    Return the plain text of a contract from a path, bytes or a raw string.

    Args:
        source (str | Path | bytes): A path to a .pdf or text file, the raw
            bytes of an uploaded file, or the contract text itself.
        filename (str | None): The original file name when ``source`` is bytes,
            used to decide whether the bytes are a PDF.

    Returns:
        str: The contract text.

    Raises:
        ValueError: If a PDF cannot be read, or appears to be an image scan.
        FileNotFoundError: If ``source`` names a file that does not exist.
    """
    # Case 1: raw bytes from an upload
    if isinstance(source, bytes):
        label = filename or "uploaded file"
        if label.lower().endswith(".pdf"):
            return _check_not_scanned(_extract_from_pdf_bytes(source, label), label)
        return source.decode("utf-8")

    # Case 2: a path on disk
    candidate = Path(source) if isinstance(source, (str, Path)) else None
    if candidate is not None and candidate.is_file():
        if candidate.suffix.lower() == ".pdf":
            return _check_not_scanned(_extract_from_pdf_path(candidate), candidate.name)
        return candidate.read_text(encoding="utf-8")

    # Case 3: a pasted string that is not a path - passthrough
    if isinstance(source, str):
        return source

    raise FileNotFoundError(
        f"extract_text received '{source}', which is neither a readable file "
        f"path, bytes nor contract text."
    )


def main():
    """Command-line check: print the first 200 characters of the sample contract."""
    sample = Path("data") / "sample_contract.txt"
    text = extract_text(sample)
    print(f"Extracted {len(text)} characters from {sample}")
    print(text[:200])


if __name__ == "__main__":
    main()