import re
from io import BytesIO, StringIO
from typing import Tuple, List
import os
from dotenv import load_dotenv
import openai

import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def extract_text_with_layout(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text_content = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text("text")

    return text_content

def extract_text_from_text_pdf(file_obj):
    text_content = extract_text_with_layout(file_obj)

    processed_data = {
        'text': text_content,
    }

    if isinstance(processed_data, dict):
        processed_data = str(processed_data)

    return processed_data

def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    text = extract_text_from_text_pdf(file)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    
    # Replace single newlines that are not followed by a space with a space
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    
    # Normalize multiple newlines to two newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E]', '', text)
    # Ensure there is a space between letters and numbers or vice versa
    text = re.sub(r'(\D)(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)(\D)', r'\1 \2', text)
    return [text], filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            chunk_doc.metadata["source"] = f"{chunk_doc.metadata['page']}-{chunk_doc.metadata['chunk']}"
            chunk_doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(chunk_doc)
    return doc_chunks

def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index

def get_index_for_text_pdf(pdf_files, pdf_names, openai_api_key=None):
    if openai_api_key is None:
        openai_api_key = st.secrets["OPENAI_API_KEY"]

    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        docs = text_to_docs(text, filename)
        index = docs_to_index(docs, openai_api_key)
        documents.append(index)
    return documents

