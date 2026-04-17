from flask import Flask, render_template, request, session
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
app.secret_key = "medibot-secret-123" # Required for session management

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load Embeddings
embeddings = download_embeddings()
index_name = "medical-chatbot"

# Vector Store
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Model (Using base gemini-2.5-flash)
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# RAG Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chatModel,
    retriever=retriever,
    output_key="answer"
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg", "")
    if not msg:
        return ""
    
    try:
        # Initialize chat history in session if not present
        if "chat_history" not in session:
            session["chat_history"] = []
        
        # Invoke chain with chat history
        response = rag_chain.invoke({
            "question": msg,
            "chat_history": session["chat_history"]
        })
        
        # Update session with new message and response
        session["chat_history"].append((msg, response["answer"]))
        session.modified = True
        
        return str(response["answer"])
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Chat error: {error_msg}")
        return error_msg

if __name__ == '__main__':
    # use_reloader=False prevents the Threading/Pickling error on Windows
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)