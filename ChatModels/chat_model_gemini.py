
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# dotenv -> Library for fetching information from the .env file
load_dotenv()

# load_dotenv() -> Loads environment variables from the .env file

# ChatGoogleGenerativeAI -> LangChain integration for Google's Gemini models
# model -> Specifies which Gemini model to use
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    max_tokens=100
)

# invoke() -> Sends the question to the LLM and returns its response
result = model.invoke("What is the capital of India?")

# .content -> Extracts the text from the returned AIMessage
print(result.content)
