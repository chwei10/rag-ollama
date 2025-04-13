"""
Configuration settings for the RAG-Ollama project
"""

import os

# Model settings
MODEL_NAME = "llama3.2:3b"

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# Text database settings
TEXT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma.db")
TEXT_COLLECTION_NAME = "restaurant_reviews"
TEXT_REVIEWS_PATH = os.path.join(DATA_DIR, "restaurant_review.txt")

# Multimodal database settings
MULTIMODAL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multimodal_chroma.db")
MULTIMODAL_COLLECTION_NAME = "multimodal_collection"

# Retrieval settings
TOP_K_RESULTS = 7  # Number of results to retrieve

# CLIP model settings
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)