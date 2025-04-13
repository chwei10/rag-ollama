"""
Multimodal RAG application for restaurant queries with image support
"""

import os
import json
import sys

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.config import MODEL_NAME, ASSETS_DIR, DATA_DIR
from src.multimodal_embeddings import retriever, add_image_document, add_text_document
from src.document_importer import process_import_command

def format_document(doc):
    """Format a document for display in the prompt"""
    doc_type = doc.metadata.get("type", "text")
    
    if doc_type == "text":
        return f"TEXT DOCUMENT:\nTitle: {doc.metadata.get('title', 'No Title')}\n" + \
               f"Rating: {doc.metadata.get('rating', 'N/A')}\n" + \
               f"Content: {doc.page_content}\n"
    
    elif doc_type == "image":
        return f"IMAGE DOCUMENT:\n" + \
               f"Caption: {doc.metadata.get('caption', 'No caption available')}\n" + \
               f"Description: {doc.metadata.get('description', 'No description available')}\n"
    
    else:
        return f"DOCUMENT ({doc_type}):\n{doc.page_content}\n"


def add_new_image(image_path, caption, metadata=None):
    """Add a new image to the database"""
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    try:
        metadata_dict = {}
        if metadata:
            if isinstance(metadata, str):
                try:
                    metadata_dict = json.loads(metadata)
                except:
                    # If not valid JSON, treat as description
                    metadata_dict = {"description": metadata}
            elif isinstance(metadata, dict):
                metadata_dict = metadata
        
        doc = add_image_document(image_path, caption, metadata_dict)
        return f"Successfully added image from {image_path} with caption: {caption}"
    except Exception as e:
        return f"Error adding image: {e}"


def add_new_text(content, metadata=None):
    """Add new text content to the database"""
    try:
        metadata_dict = {}
        if metadata:
            if isinstance(metadata, str):
                try:
                    metadata_dict = json.loads(metadata)
                except:
                    # If not valid JSON, treat as title
                    metadata_dict = {"title": metadata}
            elif isinstance(metadata, dict):
                metadata_dict = metadata
                
        doc = add_text_document(content, metadata_dict)
        return f"Successfully added text document"
    except Exception as e:
        return f"Error adding text: {e}"


def run_multimodal_app():
    """Run the multimodal RAG application"""
    model = OllamaLLM(model=MODEL_NAME)

    # Create a list to store conversation history
    conversation_history = []

    template = """
    You are an expert in answering questions about a restaurant, using both text reviews and images.

    Here are some relevant documents: {documents}

    Conversation history:
    {history}

    Here is the new question to answer: {question}

    Remember to consider our conversation history in your response.
    If the retrieved documents include images, reference the visual information in your answer.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("\nWelcome to the Multimodal Restaurant Assistant!")
    print("Ask questions about our restaurant and dishes or add new content.")
    print("\nSpecial commands:")
    print("  !add_image:path/to/image.jpg|caption|optional metadata")
    print("  !add_text:content|optional metadata")
    print("  !import_csv:path/to/file.csv")
    print("  !import_txt:path/to/file.txt")
    print("  !import_restaurant:path/to/restaurant_review.txt")
    print("  q - Quit the application\n")

    while True:
        print("\n----------------------------------")
        user_input = input("Your question/command (q to quit): ")
        if user_input.lower() == "q":
            break
        print("----------------------------------")
        
        # Check for special commands
        if user_input.startswith("!add_image:"):
            # Format: !add_image:path/to/image.jpg|caption goes here|optional metadata as json
            parts = user_input[11:].split("|")
            if len(parts) < 2:
                print("Error: Format should be !add_image:path|caption|optional metadata")
                continue
                
            image_path = parts[0].strip()
            caption = parts[1].strip()
            metadata = parts[2].strip() if len(parts) > 2 else None
            
            result = add_new_image(image_path, caption, metadata)
            print(result)
            continue
            
        elif user_input.startswith("!add_text:"):
            # Format: !add_text:content goes here|optional metadata as json
            parts = user_input[10:].split("|")
            content = parts[0].strip()
            if not content:
                print("Error: No content provided")
                continue
                
            metadata = parts[1].strip() if len(parts) > 1 else None
            
            result = add_new_text(content, metadata)
            print(result)
            continue
            
        elif user_input.startswith("!import_csv:") or user_input.startswith("!import_txt:") or user_input.startswith("!import_restaurant:"):
            # Handle all import commands
            result = process_import_command(user_input)
            print(result)
            continue
        
        # Add user question to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Format conversation history for the prompt
        formatted_history = ""
        if len(conversation_history) > 0:
            for i, message in enumerate(conversation_history):
                if i % 2 == 0:  # User message
                    formatted_history += f"User: {message.content}\n"
                else:  # AI message
                    formatted_history += f"AI: {message.content}\n"
        
        # Get relevant documents
        documents = retriever.invoke(user_input)
        
        # Format documents for the prompt
        formatted_documents = "\n".join([format_document(doc) for doc in documents])
        
        # Generate response
        result = chain.invoke({
            "documents": formatted_documents,
            "question": user_input,
            "history": formatted_history
        })
        
        # Add AI response to history
        conversation_history.append(AIMessage(content=str(result)))
        
        print(result)

    print("\nThank you for using the Multimodal Restaurant Assistant!")


if __name__ == "__main__":
    run_multimodal_app()