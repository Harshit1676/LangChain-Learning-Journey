from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

messages = [
    SystemMessage(content="Act like an python tutor"),
    HumanMessage(content="Explain builtin methods of Python")
]

result = model.invoke(messages)
messages.append(AIMessage(content="result.content"))
print(result.content)