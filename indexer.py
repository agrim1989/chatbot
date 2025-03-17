import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def load_and_split_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def index_documents(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")

def main():
    folder_path = "doc"  # Path to the folder containing PDFs
    all_documents = []
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.endswith(".pdf"):
            print(f"Processing PDF: {file_name}")
            all_documents.extend(load_and_split_pdf(file_path))
        elif file_name.endswith(".txt"):
            print(f"Processing Text: {file_name}")
            all_documents.extend(load_and_split_text(file_path))
    
    if all_documents:
        index_documents(all_documents)
        print("Indexing complete.")
    else:
        print("No valid documents found for indexing.")

if __name__ == "__main__":
    main()
