# from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# Updated import to use Ollama instead of ChatOpenAI
from langchain_community.llms import Ollama

# Load environment variables from .env
# load_dotenv()

# Updated to create an Ollama model with model="llama3:8b"
model = Ollama(model="llama3:8b")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
# HumanMessage:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
# print(f"Answer from AI: {result.content}")
print(f"1. Answer from AI: {result}")

# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
# print(f"Answer from AI: {result.content}")
print(f"2. Answer from AI: {result}")