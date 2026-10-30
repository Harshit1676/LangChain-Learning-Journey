from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from the .env file
load_dotenv()

# ChatOpenAI -> Used to interact with chat-based LLMs
# temperature -> Controls randomness; 0 makes output more deterministic
# max_tokens -> Limits the number of tokens generated in the response
model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    max_tokens=100,
    base_url="https://openrouter.ai/api/v1"
)

# Send the question to the model and get the response
result = model.invoke("Write song No one Noticed")

# Chat models return an AIMessage object; .content extracts its text
print(result.content)
