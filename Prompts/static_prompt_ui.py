
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    max_tokens=100,
    base_url="https://openrouter.ai/api/v1"
)
user_input = st.text_input("Enter your Prompt")

if st.button("Submit"):
    result = model.invoke(user_input)
    st.write(result.content)
