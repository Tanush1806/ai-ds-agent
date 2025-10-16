import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_chatbot():
    """
    Create and configure a chatbot instance with conversation memory.
    """
    # Initialize the language model
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create conversation memory
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    return conversation

def get_response(conversation, user_input):
    """
    Get response from the chatbot for the given user input.
    
    Args:
        conversation: The conversation chain instance
        user_input: User's message
        
    Returns:
        str: Chatbot's response
    """
    try:
        response = conversation.predict(input=user_input)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"
