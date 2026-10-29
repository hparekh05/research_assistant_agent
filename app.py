import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.orchestrator import orchestrator
from src.synthesizer import synthesizer_agent_stream


st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 Research Assistant Agent")
st.markdown("*Powered by Claude + Web Search + Document Analysis*")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs to include in research",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded")

        for file in uploaded_files:
            st.write(f"• {file.name}")

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown("**How it works:**")
    st.markdown("""
    1. 🌐 Web Agent searches the internet
    2. 📄 PDF Agent searches your documents
    3. 🧠 Claude reasons and synthesizes
    4. ✅ Structured answer with sources
    """)

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("web_sources") or message.get("pdf_sources"):

            with st.expander("📚 Sources"):

                if message.get("web_sources"):
                    st.markdown("**🌐 Web Sources:**")

                    for source in message["web_sources"]:
                        st.markdown(f"• {source}")

                if message.get("pdf_sources"):
                    st.markdown("**📄 Document Sources:**")

                    for source in message["pdf_sources"]:
                        st.markdown(f"• {source}")

# Chat input
if query := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.chat_history.append(HumanMessage(content=query))

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents searching..."):
            try:
                result = orchestrator(
                    query=query,
                    uploaded_files=uploaded_files,
                    chat_history=st.session_state.chat_history
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
                result = None

        if result:
            response_container = st.empty()
            full_response = ""

            words = result["answer"].split(" ")
            for word in words:
                full_response += word + " "
                response_container.markdown(full_response + "▌")
                time.sleep(0.02)

            response_container.markdown(full_response)
   
            web_sources = result.get("web_sources", [])
            pdf_sources = result.get("pdf_sources", [])

            if web_sources or pdf_sources:
                with st.expander("📚 Sources"):
                    if web_sources:
                        st.markdown("**🌐 Web Sources:**")
                        for source in web_sources:
                            st.markdown(f"- [{source}]({source})")

                    if pdf_sources:
                        st.markdown("**📄 Document Sources:**")
                        for source in pdf_sources:
                            st.markdown(f"- {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": result.get("sources", []),
                "web_sources": result.get("web_sources", []),
                "pdf_sources": result.get("pdf_sources", [])
            })
            st.session_state.chat_history.append(AIMessage(content=full_response))

st.divider()
st.caption("Research Assistant Agent | LangChain · LangGraph · Claude · Tavily · ChromaDB · MCP")