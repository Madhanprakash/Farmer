
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_project_dir() -> Path:
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    return current


PROJECT_DIR = get_project_dir()
PDF_DIR = PROJECT_DIR / "data" / "schemes"
VECTOR_DIR = PROJECT_DIR / "vectorstore"


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(
            f"No PDF files found in {PDF_DIR}. "
            "Add at least one official scheme PDF."
        )

    documents = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = pdf_path.name

        documents.extend(pages)

    print(f"Loaded {len(documents)} PDF pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise SystemExit(
            "No text extracted. The PDFs may be scanned "
            "and require OCR."
        )

    print(f"Created {len(chunks)} text chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = FAISS.from_documents(chunks, embeddings)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTOR_DIR / "tn_agri_index"))

    print(f"FAISS index saved to {VECTOR_DIR / 'tn_agri_index'}")
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
