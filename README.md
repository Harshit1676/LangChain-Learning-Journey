# LangChain Learning Journey

A collection of small Python examples documenting practical LangChain learning.

## Topics completed

- **Chat models:** invoke OpenAI-compatible models through OpenRouter, Gemini through `ChatGoogleGenerativeAI`, and hosted Hugging Face models through `HuggingFaceEndpoint` and `ChatHuggingFace`.
- **Embeddings and similarity:** create local Hugging Face and API-based embeddings, then rank sample documents with cosine similarity.
- **Prompts and Streamlit:** build and serialize a `PromptTemplate`, load it into a Streamlit UI, and submit static or selection-based prompts to a chat model.
- **Model responses:** call models with `invoke()` and read chat output from `AIMessage.content`.

These are introductory examples; they do not yet implement document retrieval, vector stores, or a complete RAG pipeline.

## Project structure

| Path | Learning example |
| --- | --- |
| `ChatModels/` | OpenRouter, Gemini, and Hugging Face chat model calls |
| `EmbeddedModels/` | Local/API embeddings and cosine similarity |
| `LLMS/` | Basic chat model invocation through OpenRouter |
| `Prompts/` | Prompt serialization, loading, and Streamlit interfaces |
| `lecture3_notes.txt` | Lecture notes |
| `requirement.txt` | Python dependencies |

## Key APIs used

`ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatHuggingFace`, `HuggingFaceEndpoint`, `OpenAIEmbeddings`, `HuggingFaceEmbeddings`, `PromptTemplate`, `load_prompt()`, `embed_query()`, `embed_documents()`, `invoke()`, and scikit-learn's `cosine_similarity`.

## Learning roadmap

Next: document loading and chunking, vector stores and retrievers, then a source-grounded RAG chain. After that, explore structured output, conversation history, tools, and evaluation.

## Setup and secrets

Install the dependencies listed in `requirement.txt`. The Streamlit interfaces and local embedding example also use Streamlit and `sentence-transformers`, which are not listed there. Configure the provider credentials required by the examples in a local `.env`, and run scripts from the project directory. Run a prompt interface with Streamlit, for example `streamlit run Prompts/static_prompt_ui.py`.

`.env`, virtual environments, Python caches, and the temporary editor runner file are excluded through `.gitignore`. Never commit credentials.
