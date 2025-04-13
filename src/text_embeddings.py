"""
Text embeddings module for the RAG-Ollama project
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from src.config import (
    MODEL_NAME, 
    TEXT_DB_PATH, 
    TEXT_COLLECTION_NAME,
    TEXT_REVIEWS_PATH,
    TOP_K_RESULTS
)

def initialize_text_embeddings():
    """Initialize text embeddings and vector store"""
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    
    # Check if we need to add documents
    add_documents = not os.path.exists(TEXT_DB_PATH)
    
    # Create vector store
    vector_store = Chroma(
        collection_name=TEXT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=TEXT_DB_PATH
    )
    
    # Load initial documents if needed
    if add_documents and os.path.exists(TEXT_REVIEWS_PATH):
        print(f"Loading reviews from {TEXT_REVIEWS_PATH}")
        df = pd.read_csv(TEXT_REVIEWS_PATH, quotechar='"')
        
        documents = []
        ids = []
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Review"], 
                metadata={
                    "title": row["Title"], 
                    "date": row["Date"], 
                    "rating": row["Rating"],
                    "type": "text"
                },
                id=str(i)
            )
            documents.append(document)
            ids.append(str(i))
            
        print(f"Adding {len(documents)} documents to vector store")
        vector_store.add_documents(documents=documents, ids=ids)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K_RESULTS
        }
    )
    
    return embeddings, vector_store, retriever

# Initialize on import
embeddings, vector_store, retriever = initialize_text_embeddings()
    
        