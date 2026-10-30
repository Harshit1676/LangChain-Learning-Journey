
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# dotenv -> Library for fetching information from the .env file
load_dotenv()

# ChatOpenAI -> Used to interact with chat models through an
# OpenAI-compatible API, including OpenRouter

model = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    temperature=0,
    max_tokens=100,
    base_url="https://openrouter.ai/api/v1"
)

# invoke() sends the question to the selected model
# and returns an AIMessage containing the response
result = model.invoke("What is the capital of India?")

# .content extracts the text from the AIMessage
print(result.content)
