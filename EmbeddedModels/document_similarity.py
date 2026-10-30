
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    dimensions=300,
    base_url="https://openrouter.ai/api/v1"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and consistency in run chases.",
    "Rohit Sharma is an Indian opening batsman famous for his elegant stroke play and double centuries in ODI cricket.",
    "MS Dhoni is a former Indian captain known for his wicketkeeping skills, calm leadership, and finishing ability.",
    "Jasprit Bumrah is an Indian fast bowler recognized for his unusual bowling action and deadly yorkers.",
    "AB de Villiers is a former South African cricketer admired for his innovative shots and versatility as a batsman."
]

query = "tell me about virat kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

result = cosine_similarity([query_embedding], doc_embeddings)[0]



index,score = sorted(list(enumerate(result)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is: ",score)