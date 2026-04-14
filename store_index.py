from dotenv import load_dotenv
import os

from src.helper import load_pdf_files, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore 
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_files("./data")
texts_chunks = text_split(extracted_data)
embeddings = download_embeddings()

#from pinecone import Pinecone
pinconee_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinconee_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    index = pc.Index(index_name)

    docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embeddings,
    index_name=index_name
)
    
