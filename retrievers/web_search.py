import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000
)

web_docs_context = """
MongoDB Atlas Vector Search allows you to build intelligent, vector-powered search applications on your Atlas data.
It integrates semantic search capabilities directly into your managed database, eliminating the need to manage a separate vector database stack.
Key Features of Atlas Vector Search:
1. Native Integration: Seamlessly search across operational and vector data in a single platform.
2. High Performance: Uses advanced Hierarchical Navigable Small World (HNSW) graphs and Inverted File (IVF) indexing for fast exact or approximate nearest neighbor search.
3. Multi-Cloud: Works across AWS, Azure, and Google Cloud platform options seamlessly.
"""

raw_documents = [Document(page_content=web_docs_context)]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)

vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user question based on the web documentation context provided:\n\n{context}"),
    ("user", "{question}")
])

web_search_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | model
    | StrOutputParser()
)