from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

parser = StrOutputParser()
class feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template='classify the sentiment of the following feedback text into positive or negative\n {feedback}\n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback\n {feedback}',
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback\n {feedback}',
    input_variables = ['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser),
    (lambda x:x.sentiment=='negative',prompt3|model|parser),
    RunnableLambda(lambda x : 'could not find any sentiment')
)
chain = classifier_chain | branch_chain

print(chain.invoke({'feedback' : 'this is the terrible phone'}))

chain.get_graph().print_ascii()