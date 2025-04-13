#!/usr/bin/env python3
"""
Main entry point for the RAG-Ollama application
"""

import argparse
import sys
import os

from src.text_app import run_text_app
from src.multimodal_app import run_multimodal_app

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description="RAG-Ollama - A RAG system for restaurant reviews using Ollama"
    )
    parser.add_argument(
        "--mode", 
        choices=["text", "multimodal"], 
        default="text",
        help="Mode to run the application in (text or multimodal)"
    )
    
    args = parser.parse_args()
    
    print("Starting RAG-Ollama application...")
    
    try:
        if args.mode == "text":
            run_text_app()
        elif args.mode == "multimodal":
            run_multimodal_app()
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        return 0
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())