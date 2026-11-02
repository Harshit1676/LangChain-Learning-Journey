from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)
prompt = PromptTemplate(
    template='Generation 5 interesting facts about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    'topic':'cricket'
})

print(result)

chain.get_graph().print_ascii()  # used to visualize the chain 