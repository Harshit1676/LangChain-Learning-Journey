from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Read the external text file
with open("sample_document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Create a Document object
document = Document(
    page_content=text,
    metadata={"source": "sample_document.txt"}
)

# Step 3: Initialize the embedding model
embedding = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Step 4: Generate embeddings
result = embedding.embed_documents([document.page_content])

# Step 5: Print results
print("Number of documents:", len(result))
print("Embedding dimensions:", len(result[0]))
print("First 10 values:", result[0][:10])