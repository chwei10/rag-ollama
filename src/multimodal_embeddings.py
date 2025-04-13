"""
Multimodal embeddings module for the RAG-Ollama project
"""

from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from PIL import Image
import base64
import io
import numpy as np

from src.config import (
    MODEL_NAME,
    MULTIMODAL_DB_PATH,
    MULTIMODAL_COLLECTION_NAME,
    TEXT_REVIEWS_PATH,
    TOP_K_RESULTS,
    CLIP_MODEL_ID
)

# Try to import torch and transformers, but provide fallback
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAVE_TORCH = True
    print("PyTorch and Transformers available - full multimodal capabilities enabled")
except ImportError:
    print("Warning: PyTorch or Transformers not available. Image processing will use text-only fallback.")
    HAVE_TORCH = False

# Base embeddings for text
text_embeddings = OllamaEmbeddings(model=MODEL_NAME)

# Initialize CLIP model for image embeddings
if HAVE_TORCH:
    try:
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        print(f"Successfully loaded CLIP model: {CLIP_MODEL_ID}")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Proceeding with text-only embeddings")
        clip_model = None
        clip_processor = None
else:
    clip_model = None
    clip_processor = None

class MultiModalEmbeddings:
    """
    MultiModalEmbeddings class that handles both text and image embeddings
    """
    def __init__(self):
        self.text_embeddings = text_embeddings
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        
    def _embed_from_caption(self, doc, embeddings_list):
        """Helper method to embed from caption when image embedding fails"""
        # This is now handled directly in embed_documents
        # Kept for backward compatibility
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        if "caption" in metadata:
            text_emb = self.text_embeddings.embed_documents([metadata["caption"]])[0]
            embeddings_list.append(text_emb)
        else:
            # Use zeros as fallback
            embeddings_list.append([0.0] * 512)
        
    def embed_documents(self, documents):
        """
        Embed a list of documents, handling different modalities
        """
        embeddings_list = []
        
        for doc in documents:
            # Check if this is a Document object or a string
            if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                # It's a Document object
                doc_content = doc.page_content
                doc_type = doc.metadata.get("type", "text")
                doc_metadata = doc.metadata
            else:
                # It's a string (text content)
                doc_content = doc
                doc_type = "text"
                doc_metadata = {}
            
            if doc_type == "text":
                # Use text embeddings
                text_emb = self.text_embeddings.embed_documents([doc_content])[0]
                embeddings_list.append(text_emb)
                
            elif doc_type == "image":
                if self.clip_model is not None and HAVE_TORCH:
                    # Get image data from content (stored as base64)
                    try:
                        image_data = base64.b64decode(doc_content)
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Process image with CLIP
                        inputs = self.clip_processor(
                            text=None,
                            images=image, 
                            return_tensors="pt", 
                            padding=True
                        )
                        
                        with torch.no_grad():
                            image_features = self.clip_model.get_image_features(**inputs)
                            
                        # Convert to list and normalize
                        image_emb = image_features[0].cpu().numpy().tolist()
                        embeddings_list.append(image_emb)
                    except Exception as e:
                        print(f"Error embedding image: {e}")
                        # Fall back to caption/alt-text
                        if "caption" in doc_metadata:
                            text_emb = self.text_embeddings.embed_documents([doc_metadata["caption"]])[0]
                            embeddings_list.append(text_emb)
                        else:
                            # Use zeros as fallback
                            embeddings_list.append([0.0] * 512)
                else:
                    # No CLIP model available, fall back to caption
                    if "caption" in doc_metadata:
                        text_emb = self.text_embeddings.embed_documents([doc_metadata["caption"]])[0]
                        embeddings_list.append(text_emb)
                    else:
                        # Use zeros as fallback
                        embeddings_list.append([0.0] * 512)
                        
            elif doc_type == "audio":
                # For future implementation
                # Use caption/transcript if available
                if "transcript" in doc_metadata:
                    text_emb = self.text_embeddings.embed_documents([doc_metadata["transcript"]])[0]
                    embeddings_list.append(text_emb)
                else:
                    # Use zeros as fallback
                    embeddings_list.append([0.0] * 512)
            
            else:
                # Default to text embedding for unknown types
                text_emb = self.text_embeddings.embed_documents([doc_content])[0]
                embeddings_list.append(text_emb)
                
        return embeddings_list
    
    def embed_query(self, query):
        """
        Embed a query (text only for now)
        """
        return self.text_embeddings.embed_query(query)


def initialize_multimodal_embeddings():
    """Initialize multimodal embeddings and vector store"""
    # Check if we need to add documents
    add_documents = not os.path.exists(MULTIMODAL_DB_PATH)

    # Initialize multimodal embeddings
    mm_embeddings = MultiModalEmbeddings()

    # Create vector store
    vector_store = Chroma(
        collection_name=MULTIMODAL_COLLECTION_NAME,
        embedding_function=mm_embeddings,
        persist_directory=MULTIMODAL_DB_PATH
    )

    # Load initial documents if needed
    if add_documents and os.path.exists(TEXT_REVIEWS_PATH):
        # Load text documents from existing reviews
        print(f"Adding text documents to multimodal vector store from {TEXT_REVIEWS_PATH}...")
        df = pd.read_csv(TEXT_REVIEWS_PATH, quotechar='"')
        documents = []
        
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Review"],
                metadata={
                    "title": row["Title"], 
                    "date": row["Date"], 
                    "rating": row["Rating"],
                    "type": "text"
                }
            )
            documents.append(document)
        
        vector_store.add_documents(documents=documents)
        print(f"Added {len(documents)} text documents to multimodal vector store")

    # Set up retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K_RESULTS
        }
    )
    
    return mm_embeddings, vector_store, retriever


# Function to add a text document
def add_text_document(text, metadata=None):
    """Add a text document to the multimodal vector store"""
    if metadata is None:
        metadata = {}
    
    metadata["type"] = "text"
    
    doc = Document(
        page_content=text,
        metadata=metadata
    )
    
    vector_store.add_documents([doc])
    return doc


# Function to add an image
def add_image_document(image_path, caption=None, metadata=None):
    """Add an image document to the multimodal vector store"""
    if metadata is None:
        metadata = {}
    
    metadata["type"] = "image"
    
    if caption:
        metadata["caption"] = caption
    
    # Read image and convert to base64
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    doc = Document(
        page_content=image_data,
        metadata=metadata
    )
    
    vector_store.add_documents([doc])
    return doc


# Initialize on import
mm_embeddings, vector_store, retriever = initialize_multimodal_embeddings()