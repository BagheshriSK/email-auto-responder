import argparse
import os
import shutil
import base64
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from main import authenticate_gmail


# Constants
CHROMA_PATH = "chroma"
DATA_FOLDER = "data"  # Folder where PDF, DOCX, and TXT files are stored
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def fetch_replies(service) -> List[Document]:
    """
    Fetch all replies from the email account and return as documents.
    """
    print("Fetching replies from secondary email account...")
    results = service.users().messages().list(userId='me', q="in:sent").execute()
    messages = results.get('messages', [])

    documents = []

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        # Extract subject and sender
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
        
        # Skip messages without body content
        parts = payload.get('parts', [])
        body = ''
        if parts:
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    body = base64.urlsafe_b64decode(part.get('body', {}).get('data', '')).decode('utf-8')
                    break
        else:
            body = msg.get('snippet', '')

        # Add the reply to the document list
        documents.append(Document(
            page_content=body,
            metadata={
                "subject": subject,
                "sender": sender,
                "email_id": message['id']
            }
        ))

    return documents

def load_documents_from_directory(directory: str) -> List[Document]:
    """
    Load documents from a directory, supporting PDF, DOCX, and TXT files using LangChain document loaders.
    """
    documents = []
    # Traverse the directory for files
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix == ".pdf":
            # Load PDF document using PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
            print(f"Loaded PDF file: {file_path}")
        elif file_path.suffix == ".docx":
            # Load DOCX document using Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
            documents.extend(loader.load())
            print(f"Loaded DOCX file: {file_path}")
        elif file_path.suffix == ".txt":
            # Load TXT document using TextLoader
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
            print(f"Loaded TXT file: {file_path}")
        else:
            print(f"Skipping unsupported file type: {file_path}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using a text splitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len)
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Add unique IDs to document chunks for tracking.
    """
    source_chunk_count = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        index = source_chunk_count.get(source, 0)
        chunk.metadata["id"] = f"{source}:{index}"
        source_chunk_count[source] = index + 1
    return chunks

def get_embedding_function():
    """
    Returns the embedding function for Chroma database.
    """
    return OllamaEmbeddings(model="nomic-embed-text")

def add_to_chroma(chunks: List[Document]):
    """
    Add document chunks to Chroma database if not already present.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calculate unique chunk IDs
    chunks = calculate_chunk_ids(chunks)

    # Filter out existing documents
    existing_ids = set(db.get()['ids']) if db.get()['ids'] else set()

    # Filter out the chunks that already exist
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new documents to Chroma.")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    else:
        print("No new documents to add.")

def clear_database():
    """
    Clear the Chroma database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared.")
    else:
        print("No database found to clear.")

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Process and manage cold email data.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    # Load documents from all PDF, DOCX, and TXT files in the data folder
    file_documents = load_documents_from_directory(DATA_FOLDER)
    
    if not file_documents:
        print(f"No PDF, DOCX, or TXT files found in the {DATA_FOLDER} folder.")
    
    # Authenticate with secondary email account
    service = authenticate_gmail("secondary")
    
    # Fetch all replies
    email_documents = fetch_replies(service)

    # Split documents into smaller chunks
    chunks = split_documents(file_documents + email_documents)
    
    # Add chunks to the Chroma database
    add_to_chroma(chunks)
    print("Added all the documents successfully")

if __name__ == "__main__":
    main()
