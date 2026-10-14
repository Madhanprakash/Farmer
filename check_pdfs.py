
from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/schemes")

pdf_files = list(PDF_DIR.glob("*.pdf"))

if not pdf_files:
    print(f"No PDFs found in: {PDF_DIR.resolve()}")
    raise SystemExit(1)

for pdf_path in pdf_files:
    print(f"\nFile: {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))
        print(f"Pages: {len(reader.pages)}")

        extracted_pages = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                extracted_pages += 1

        print(f"Pages with extractable text: {extracted_pages}")

        if extracted_pages == 0:
            print("WARNING: May require OCR.")
        elif extracted_pages < len(reader.pages):
            print("WARNING: Some pages may require OCR.")
        else:
            print("Basic text extraction check passed.")

    except Exception as exc:
        print(f"ERROR reading PDF: {exc}")
