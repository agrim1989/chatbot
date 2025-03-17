import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Google API Key
os.environ["GOOGLE_API_KEY"] = ""

# Function to load and split text files
def load_and_split_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Function to load and split PDFs
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Function to index documents using FAISS
def index_documents(file_name, documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(f"faiss_index_{file_name}")
    print(f"Indexing complete. FAISS index saved for {file_name}.")

# Function to process and index all files in the ./doc folder
def process_all_files():
    folder_path = "./doc"
    indexed_files = {}
    
    if not os.path.exists(folder_path):
        print("Folder ./doc does not exist.")
        return indexed_files
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Skip non-text and non-PDF files
        if not (file_name.endswith(".pdf") or file_name.endswith(".txt")):
            continue
        
        # Skip already indexed files
        if os.path.exists(f"faiss_index_{file_name}"):
            print(f"Skipping already indexed file: {file_name}")
            continue
        
        # Load and index the file
        documents = load_and_split_pdf(file_path) if file_name.endswith(".pdf") else load_and_split_text(file_path)
        
        if documents:
            index_documents(file_name, documents)
            indexed_files[file_name] = documents
    
    return indexed_files

# Load FAISS index for a specific file
def load_faiss_index(file_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(f"faiss_index_{file_name}", embeddings, allow_dangerous_deserialization=True)

# Streamlit UI
def chatbot_ui():
    st.title("Chat with Your Documents")
    st.write("Ask questions based on the indexed files in ./doc folder.")
    
    indexed_files = process_all_files()
    file_list = [f for f in os.listdir("./doc") if f.endswith(".pdf") or f.endswith(".txt")]
    
    if not file_list:
        st.error("No indexed files available.")
        return
    
    selected_file = st.selectbox("Select a document:", file_list)
    
    if selected_file:
        try:
            vectorstore = load_faiss_index(selected_file)
            retriever = vectorstore.as_retriever()
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp")
            
            # Corrected RetrievalQA initialization
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            
            user_input = st.text_input("Ask a question:")
            if user_input:
                response = qa.run(user_input)
                st.write("### Answer:")
                st.write(response)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")

if __name__ == "__main__":
    chatbot_ui()
