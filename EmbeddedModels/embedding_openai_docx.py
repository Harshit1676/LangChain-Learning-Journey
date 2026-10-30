
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

# dotenv -> Library for fetching information from the .env file
load_dotenv()

# OpenAIEmbeddings -> Converts text into numerical vectors
# base_url -> Routes the request to OpenRouter instead of OpenAI
embedding = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

documents = ["Delhi is the Capital of India",
             "Kolkata is the Capital of West Bengal",
             "Paris is the capital of france"]
# embed_query() -> Converts a single text query into an embedding vector
result = embedding.embed_documents(documents)


# Prints the embedding vector's dimensions
print("Vector dimensions:", len(result))

print(str(result))
