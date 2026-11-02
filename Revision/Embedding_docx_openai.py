from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
    dimensions=32
)
documents = ["Supervised Learning: The model learns from labelled data. Examples include spam detection, house price prediction, and disease classification.",
             "Unsupervised Learning: The model learns patterns from unlabelled data. Examples include customer segmentation and clustering.",
             "Reinforcement Learning: The model learns through interaction with an environment using rewards and penalties. Examples include robotics and game-playing systems"]
result = embedding.embed_documents(documents)

print(result)

print("Vector size: ",len(result))