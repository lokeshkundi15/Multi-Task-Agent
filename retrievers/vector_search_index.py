import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# OpenRouter Configuration
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000
)

DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

client = MongoClient(os.getenv("ATLAS_CONNECTION_STRING"))
collection = client[DB_NAME][COLLECTION_NAME]

# Local PDF Absolute Path (Fixed)
loader = PyPDFLoader(r"D:\COURSES\RoadMap\.4 Level Up LLM App Develoent with LangChain and OpenAI\~Get Your Files Here !\9 - LLM Fine-Tuning with the OpenAI Tools and Functions\Multi-Task-Agent\AI_Job_Roadmap.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)
vector_store.add_documents(docs)

retriever = vector_store.as_retriever(
   search_type="similarity",
   search_kwargs={"k": 2}
)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, use your own knowledge.

{context}

Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | model
    | StrOutputParser()
)