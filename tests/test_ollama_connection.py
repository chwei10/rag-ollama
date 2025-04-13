import requests
import sys

# Test basic connection to Ollama API
try:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "llama3.2:3b", "prompt": "Hello world"}
    )
    if response.status_code == 200:
        print("✅ Successfully connected to Ollama API")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ API returned status code {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")
    
# Try with langchain_ollama
try:
    from langchain_ollama import OllamaEmbeddings
    
    print("\nTesting OllamaEmbeddings...")
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    result = embeddings.embed_query("Hello world")
    
    print(f"✅ Successfully got embeddings with {len(result)} dimensions")
except Exception as e:
    print(f"❌ OllamaEmbeddings error: {e}")
    import traceback
    traceback.print_exc()

print("\nSystem information:")
print(f"Python version: {sys.version}")