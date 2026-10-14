
from pathlib import Path
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
PDF_DIR = PROJECT_DIR / "data" / "schemes"
VECTOR_DIR = PROJECT_DIR / "vectorstore"
INDEX_NAME = "tn_agri_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("=" * 60)
    print("Tamil Nadu Agricultural Scheme - RAG Ingestion")
    print("=" * 60)

    print(f"Project directory : {PROJECT_DIR}")
    print(f"PDF directory     : {PDF_DIR}")
    print(f"Vector directory  : {VECTOR_DIR}")


    if not PDF_DIR.exists():
        print("\nERROR: PDF directory does not exist.")
        print(f"Create this directory:\n{PDF_DIR}")
        sys.exit(1)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("\nERROR: No PDF files found.")
        print("Place your official scheme PDFs in:")
        print(PDF_DIR)
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF file(s).")


    documents = []
    failed_files = []

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            if not pages:
                print("WARNING: No pages extracted.")
                failed_files.append(pdf_path.name)
                continue

            for page in pages:
                page.metadata["source_file"] = pdf_path.name
                page.metadata["document_type"] = "agricultural_scheme"

            documents.extend(pages)

            print(f"Successfully loaded {len(pages)} page(s).")

        except Exception as exc:
            print(f"ERROR reading {pdf_path.name}: {exc}")
            failed_files.append(pdf_path.name)

    if not documents:
        print("\nERROR: No PDF pages were loaded successfully.")
        sys.exit(1)

    print(f"\nTotal pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Remove chunks containing only whitespace
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        print("\nERROR: No text chunks created.")
        print("Your PDFs may contain scanned images.")
        print("OCR may be required to extract their text.")
        sys.exit(1)

    print(f"Total text chunks created: {len(chunks)}")


    print("\nLoading embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        print(f"\nERROR loading embedding model: {exc}")
        print("Check your package installation and internet connection.")
        sys.exit(1)


    print("\nCreating FAISS vector index...")

    try:
        store = FAISS.from_documents(chunks, embeddings)

        VECTOR_DIR.mkdir(parents=True, exist_ok=True)

        index_path = VECTOR_DIR / INDEX_NAME
        store.save_local(str(index_path))

    except Exception as exc:
        print(f"\nERROR creating or saving FAISS index: {exc}")
        sys.exit(1)


    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"PDF files discovered : {len(pdf_files)}")
    print(f"Pages loaded         : {len(documents)}")
    print(f"Text chunks created  : {len(chunks)}")
    print(f"FAISS index saved to : {index_path}")

    if failed_files:
        print("\nWARNING: Some files failed to load:")
        for filename in failed_files:
            print(f"- {filename}")

    print("\nNext step: connect this index to your RAG retrieval chain.")


if __name__ == "__main__":
    main()
