
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# dotenv -> Library for fetching information from the .env file
load_dotenv()

# os.getenv() -> Reads the Hugging Face token from environment variables
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# HuggingFaceEndpoint -> Connects to a hosted Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

# ChatHuggingFace -> Wraps the LLM in a chat-model interface
model = ChatHuggingFace(llm=llm)

# invoke() -> Sends the prompt to the model
result = model.invoke("What is the capital of India?")

# .content -> Extracts the generated text
print(result.content)
