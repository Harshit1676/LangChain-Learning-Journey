import ast

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chat_template = ChatPromptTemplate([
    ('system','you are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
# load chat history
with open('Prompts/chat_history.txt') as f:
    for line in f:
        message_type, _, content = line.partition('(content=')
        if content:
            role = 'human' if message_type.startswith('HumanMessage') else 'ai'
            chat_history.append((role, ast.literal_eval(content.rstrip().removesuffix(')'))))

print(chat_history)

#create prompt

prompt = chat_template.invoke({
    'chat_history':chat_history, 
    'query':'Where is my refund'
    })

print(prompt)
