from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate a name, age and city of a fictional {place} person\n'
    '{format_instruction}',
    input_variables=['place'],
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)

# prompt = template.invoke({'place': 'indian'})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
print(chain)
final_result = chain.invoke({'place':'USA'})
print(final_result)