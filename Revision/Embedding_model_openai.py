from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=32
)

# if we don't provide the dimension it will create vector of 3072 which is default for the larger embedding model of openai

result = embedding.embed_query("I want to become a software engineer")

print(result)
print("Vector dimensions:", len(result))