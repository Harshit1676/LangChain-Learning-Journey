
from langchain_openai import ChatOpenAI

# dotenv -> It is the library for fetching information from the .env file
from dotenv import load_dotenv

# load_dotenv() -> It loads the environment variables present in the .env file
load_dotenv()
# ChatOpenAI -> It is used to interact with chat-based LLM models
# base_url -> It specifies the API endpoint of OpenRouter instead of OpenAI
# api_key -> It reads the API key from the OPENAI_API_KEY environment variable
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
)

# The question inside invoke() will go to the LLM model above,
# and it will return the model's response
result = llm.invoke("What is the capital of India?")

# Chat models return an AIMessage object, so .content extracts the text
print(result.content)

# LLMs take string input and return string output;
# chat models return a message object containing the response
