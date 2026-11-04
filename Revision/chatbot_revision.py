from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

messages = [
    SystemMessage(content='You are tutor')
]

while True:
    user_input = input('user_input:')
    if user_input.lower == 'exit':
        break;
    messages.append(HumanMessage(content=user_input))
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print('AI: ',result.content)

print(messages)

