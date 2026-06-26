from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from retrievers.vector_search_index import rag_chain
from retrievers.web_search import web_search_chain

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000
)  

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool 
def vector_search_query(query: str) -> str:
    """search queries using vector search"""
    return rag_chain.invoke(query)

@tool
def web_search_query(query: str) -> str:
    """search queries using web search for additional information"""
    return web_search_chain.invoke(query)

tools = {
    "get_word_length": get_word_length,
    "vector_search_query": vector_search_query,
    "web_search_query": web_search_query
}

llm_with_tools = llm.bind_tools(list(tools.values()))

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very powerful assistant. First, try to use vector_search_query or web_search_query to find specific data.
            If the tools do not return enough context, use your internal knowledge to give a complete answer.""",
        ),
        ("user", "{input}"),
    ]
)

def run_agent(query):
    # 1. First Priority: Try Local Database PDF
    try:
        db_output = tools["vector_search_query"].invoke(query)
        if db_output and "don't know" not in db_output.lower() and len(db_output.strip()) > 50:
            final_prompt = f"Question: {query}\nContext from Database: {db_output}\nGive a final perfect answer using this context."
            return llm.invoke(final_prompt).content
    except Exception:
        pass

    # 2. Second Priority: Try Web Documentation
    try:
        web_output = tools["web_search_query"].invoke(query)
        if web_output and "don't know" not in web_output.lower() and len(web_output.strip()) > 50:
            final_prompt = f"Question: {query}\nContext from Web: {web_output}\nGive a final perfect answer using this web context."
            return llm.invoke(final_prompt).content
    except Exception:
        pass

    # 3. Fallback: General LLM Knowledge
    chain = prompt | llm_with_tools
    response = chain.invoke({"input": query})
    return response.content