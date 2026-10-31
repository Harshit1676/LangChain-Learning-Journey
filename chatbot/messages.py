from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

messages = [
    SystemMessage(content='You are a helpful assitant'),
    HumanMessage(content='Tell me about Langchain'),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)