# RAG-Ollama: Multimodal Retrieval Augmented Generation

A powerful Retrieval Augmented Generation (RAG) system for restaurant reviews that supports both text and images using the Ollama local LLM server.

## Features

- **Text-based RAG**: Query restaurant reviews with natural language questions
- **Multimodal RAG**: Add images with captions to enhance the knowledge base
- **Conversation Memory**: Maintains context across multiple turns of dialog
- **Easy Content Addition**: Simple commands to add new text and images
- **Local Processing**: All processing happens locally via Ollama

## Project Structure

```
rag-ollama/
├── assets/            # Directory for image assets
├── data/              # Dataset directory
│   └── restaurant_review.txt    # Restaurant review dataset
├── main.py            # Main entry point script
├── requirements.txt   # Python dependencies
├── src/               # Source code
│   ├── config.py      # Configuration settings
│   ├── multimodal_app.py        # Multimodal application
│   ├── multimodal_embeddings.py # Multimodal embedding system
│   ├── text_app.py    # Text-only application
│   └── text_embeddings.py       # Text embedding system
└── tests/             # Test scripts
```

## Setup

1. Ensure you have Ollama installed and running:
   ```
   # Check if Ollama is installed
   ollama --version
   
   # Start the Ollama server if needed
   ollama serve
   ```

2. Pull the required model:
   ```
   ollama pull llama3.2:3b
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   # Text-only mode
   python main.py --mode text
   
   # Multimodal mode
   python main.py --mode multimodal
   ```

## Usage

### Text-Only Mode

In text mode, you can ask questions about the restaurant based on reviews:

```
Your question (q to quit): What do people say about the pizza crust?
```

### Multimodal Mode

In multimodal mode, you have additional commands:

1. Ask questions about the restaurant:
   ```
   Your question/command (q to quit): Are there any good vegetarian options?
   ```

2. Add a new image:
   ```
   Your question/command (q to quit): !add_image:assets/margherita.jpg|Delicious margherita pizza with fresh basil
   ```

3. Add a new text review:
   ```
   Your question/command (q to quit): !add_text:The pizza was excellent with a perfectly crispy crust.|Great First Visit
   ```

## How It Works

1. **Vector Database**: Both text and images are stored in a ChromaDB vector database
2. **Embeddings**:
   - Text is embedded using the Ollama model
   - Images are embedded using CLIP (Contrastive Language-Image Pre-training) or captions as fallback
3. **Retrieval**: When you ask a question, the system finds the most semantically similar content
4. **Generation**: The LLM generates a response based on the retrieved context and conversation history

## Requirements

- Python 3.8+
- Ollama
- Basic Python packages (see requirements.txt)
- Optional: PyTorch and Transformers for full multimodal capabilities

## Future Enhancements

- Audio processing support
- Video understanding
- More advanced conversation capabilities
- Integration with menus and ordering systems