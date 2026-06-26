import streamlit as st
from agent import run_agent


st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Source Autonomous Agent")
st.markdown("Ask anything from your Private Data (MongoDB) or Web Docs (Chroma)!")
st.write("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Type your question here..."):

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})


    with st.chat_message("assistant"):
        with st.spinner("Agent is searching across sources..."):
            try:
                response = run_agent(user_query)
                st.markdown(response)
            except Exception as e:
                response = f"⚠️ Error running agent: {str(e)}"
                st.markdown(response)
                
    st.session_state.messages.append({"role": "assistant", "content": response})