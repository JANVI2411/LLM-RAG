import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from unstructured.partition.pdf import partition_pdf

def partition_document(pdf_path: str):
    return partition_pdf(
        pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=400,
        combine_text_under_n_chars=50,
        new_after_n_chars=200
    )

def pdf_parser(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    return docs

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
