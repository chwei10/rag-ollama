"""
Text-only RAG application for pizza restaurant queries
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.config import MODEL_NAME
from src.text_embeddings import retriever

def run_text_app():
    """Run the text-only RAG application"""
    model = OllamaLLM(model=MODEL_NAME)

    # Create a list to store conversation history
    conversation_history = []

    template = """
    You are an expert in answering questions about a pizza restaurant

    Here are some relevant reviews: {reviews}

    Conversation history:
    {history}

    Here is the new question to answer: {question}

    Remember to consider our conversation history in your response.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("\nWelcome to the Pizza Restaurant Assistant!")
    print("Ask questions about our restaurant and dishes. Type 'q' to quit.\n")

    while True:
        print("\n----------------------------------")
        question = input("Your question (q to quit): ")
        if question.lower() == "q":
            break
        print("----------------------------------")
        
        # Add user question to history
        conversation_history.append(HumanMessage(content=question))
        
        # Format conversation history for the prompt
        formatted_history = ""
        if len(conversation_history) > 0:
            for i, message in enumerate(conversation_history):
                if i % 2 == 0:  # User message
                    formatted_history += f"User: {message.content}\n"
                else:  # AI message
                    formatted_history += f"AI: {message.content}\n"
        
        # Get relevant reviews
        reviews = retriever.invoke(question)
        
        # Generate response
        result = chain.invoke({
            "reviews": reviews,
            "question": question,
            "history": formatted_history
        })
        
        # Add AI response to history
        conversation_history.append(AIMessage(content=str(result)))
        
        print(result)

    print("\nThank you for using the Pizza Restaurant Assistant!")

if __name__ == "__main__":
    run_text_app()
