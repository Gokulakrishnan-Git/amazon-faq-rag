import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# -------------------------
# Initialize models and database
# -------------------------
@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Conversational RAG chain (maintains memory)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


qa_chain = load_pipeline()

# -------------------------
# Streamlit Chat UI
# -------------------------
st.set_page_config(page_title="Amazon FAQ Chatbot", page_icon="🛍️")
st.title("🛍️ Amazon FAQ Chatbot (Conversational RAG + Groq)")

st.markdown(
    "Ask anything about Amazon products, orders, or returns. The chatbot will respond conversationally."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain(
                {"question": prompt, "chat_history": st.session_state.chat_history}
            )
            answer = result["answer"]
            st.markdown(answer)

            # Show retrieved context (optional)
            with st.expander("📚 Retrieved Context"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.write(f"**[{i}]** {doc.page_content[:400]}...")

            # Save assistant reply
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append((prompt, answer))

# Sidebar
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This chatbot uses **Retrieval-Augmented Generation (RAG)** with a FAISS vector store "
    "and **Groq LLaMA 3.1 8B Instant** for conversational responses. "
    "It maintains chat memory and answers Amazon FAQ-style questions naturally."
)
