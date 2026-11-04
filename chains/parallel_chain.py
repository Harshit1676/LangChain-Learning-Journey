
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

model2 = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text:\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question-answer pairs from the following text:\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='''
Combine the following notes and quiz into one well-organized document.
NOTES:
{notes}
QUIZ:
{quiz}
''',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods
used for classification, regression and outlier detection.

Advantages:
- Effective in high-dimensional spaces.
- Can work when dimensions exceed the number of samples.
- Uses support vectors, making it memory efficient.
- Supports different kernel functions.

Disadvantages:
- Kernel and regularization selection are important when features
  greatly outnumber samples.
- Does not directly provide probability estimates.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()