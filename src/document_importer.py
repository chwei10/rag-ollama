"""
Document importer module for importing various file types into the RAG system
"""

import os
import json
import sys
import pandas as pd
import csv
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Tuple, Union

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR
from src.multimodal_embeddings import add_text_document, add_image_document


def import_from_csv(file_path: str) -> Tuple[int, str]:
    """
    Import reviews from a CSV file into the database
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple containing (number of documents added, status message)
    """
    if not os.path.exists(file_path):
        return 0, f"Error: File not found at {file_path}"
    
    try:
        # Try to detect the CSV format
        with open(file_path, 'r', encoding='utf-8') as file:
            dialect = csv.Sniffer().sniff(file.read(4096))
            file.seek(0)
            has_header = csv.Sniffer().has_header(file.read(4096))
            file.seek(0)
            
            # Read the CSV file
            reader = csv.reader(file, dialect)
            
            # Get headers if they exist
            headers = next(reader) if has_header else None
            
            # Prepare for adding documents
            added_count = 0
            
            # Process based on whether we have headers
            for row in reader:
                if headers:
                    # Try to find common review-related columns
                    content_col = next((i for i, h in enumerate(headers) if h.lower() in 
                                   ['review', 'content', 'text', 'comment', 'feedback']), -1)
                    title_col = next((i for i, h in enumerate(headers) if h.lower() in 
                                  ['title', 'subject', 'name']), -1)
                    rating_col = next((i for i, h in enumerate(headers) if h.lower() in 
                                   ['rating', 'score', 'stars']), -1)
                    date_col = next((i for i, h in enumerate(headers) if h.lower() in 
                                  ['date', 'time', 'datetime']), -1)
                    
                    # Create metadata
                    metadata = {}
                    if title_col >= 0 and title_col < len(row):
                        metadata["title"] = row[title_col]
                    if rating_col >= 0 and rating_col < len(row):
                        metadata["rating"] = row[rating_col]
                    if date_col >= 0 and date_col < len(row):
                        metadata["date"] = row[date_col]
                    
                    # Add the document if we found content
                    if content_col >= 0 and content_col < len(row) and row[content_col].strip():
                        add_text_document(row[content_col], metadata)
                        added_count += 1
                    else:
                        # If we couldn't identify a clear content column, use the whole row
                        content = ", ".join(row)
                        add_text_document(content, {})
                        added_count += 1
                else:
                    # No headers, assume the entire row is content
                    content = ", ".join(row)
                    add_text_document(content, {})
                    added_count += 1
            
            return added_count, f"Successfully imported {added_count} documents from {file_path}"
    except Exception as e:
        return 0, f"Error importing from CSV: {e}"


def import_from_restaurant_txt(file_path: str) -> Tuple[int, str]:
    """
    Import reviews from the restaurant_review.txt format into the database
    
    Args:
        file_path: Path to the file in restaurant_review.txt format
        
    Returns:
        Tuple containing (number of documents added, status message)
    """
    if not os.path.exists(file_path):
        return 0, f"Error: File not found at {file_path}"
    
    try:
        df = pd.read_csv(file_path, quotechar='"')
        added_count = 0
        
        for _, row in df.iterrows():
            metadata = {
                "title": row.get("Title", "No Title"),
                "date": row.get("Date", "Unknown Date"),
                "rating": row.get("Rating", "No Rating"),
                "type": "text"
            }
            
            add_text_document(row["Review"], metadata)
            added_count += 1
        
        return added_count, f"Successfully imported {added_count} documents from {file_path}"
    except Exception as e:
        return 0, f"Error importing restaurant reviews: {e}"


def import_from_txt(file_path: str) -> Tuple[int, str]:
    """
    Import content from a plain text file into the database
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Tuple containing (number of documents added, status message)
    """
    if not os.path.exists(file_path):
        return 0, f"Error: File not found at {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Check if it looks like the restaurant_review.txt format
        if content.startswith("Title,Date,Rating,Review"):
            return import_from_restaurant_txt(file_path)
            
        # Otherwise, treat as plain text
        # Split by lines or paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # If no paragraphs, try lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            if not lines:
                return 0, "Error: No content found in the file"
                
            # Add each line as a separate document
            for i, line in enumerate(lines):
                add_text_document(line, {"title": f"Line {i+1}", "source": os.path.basename(file_path)})
                
            return len(lines), f"Successfully imported {len(lines)} lines from {file_path}"
        else:
            # Add each paragraph as a separate document
            for i, paragraph in enumerate(paragraphs):
                add_text_document(paragraph, {"title": f"Paragraph {i+1}", "source": os.path.basename(file_path)})
                
            return len(paragraphs), f"Successfully imported {len(paragraphs)} paragraphs from {file_path}"
    except Exception as e:
        return 0, f"Error importing from text file: {e}"


def process_import_command(command: str) -> str:
    """
    Process an import command from the user interface
    
    Args:
        command: The user command
        
    Returns:
        Status message
    """
    if command.startswith("!import_csv:"):
        # Format: !import_csv:path/to/file.csv
        file_path = command[12:].strip()
        _, message = import_from_csv(file_path)
        return message
        
    elif command.startswith("!import_txt:"):
        # Format: !import_txt:path/to/file.txt
        file_path = command[12:].strip()
        _, message = import_from_txt(file_path)
        return message
        
    elif command.startswith("!import_restaurant:"):
        # Format: !import_restaurant:path/to/file.txt
        file_path = command[19:].strip()
        _, message = import_from_restaurant_txt(file_path)
        return message
        
    return "Unknown import command format"