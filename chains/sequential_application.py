from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

prompt1 = PromptTemplate(
    template = 'Give a detailed report on {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate a 5 pointer summary from the following text\n {text}',
    input_variables = ['text']
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({
    'topic' : 'Unemployment in india'
})
print(prompt1)
print(result)

chain.get_graph().print_ascii()