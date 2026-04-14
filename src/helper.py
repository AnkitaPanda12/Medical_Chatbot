from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

#Extracting data from the pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf",
        loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20,
        )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
        )
    return embeddings

embeddings = download_embeddings()