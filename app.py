import os
import streamlit as st
from rag_chatbot import RAGChatbot
from config import CHROMA_DB_PATH

st.set_page_config(page_title="NexaCore Assistant", page_icon="🤖")
st.title("🤖 NexaCore Knowledge Assistant")
st.caption("Ask questions about the NexaCore reference library (fictional training data).")


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_bot():
    bot = RAGChatbot()
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        bot.create_vector_store()
    else:
        bot.load_documents()
        chunks = bot.chunk_documents()
        bot.create_vector_store(chunks)
    bot.build_qa_chain()
    return bot


bot = get_bot()

with st.sidebar:
    st.header("About")
    st.write("RAG chatbot: local embeddings + Ollama LLM + ChromaDB.")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(s)

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, docs = bot.ask(question)
                sources = [
                    f"Page {d.metadata.get('page', 0) + 1}: {d.page_content[:150]}..."
                    for d in docs
                ]
            except Exception as e:
                answer, sources = f"Error: {e}", []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.caption(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )