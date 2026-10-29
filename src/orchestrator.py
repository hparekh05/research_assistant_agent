from langgraph.graph import StateGraph, END
# StateGraph builds the agent workflow as a graph
# END is a special marker that says "stop here"
from langgraph.prebuilt import ToolNode
# Pre-built node that handles tool execution automatically
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
# Decorator that turns a Python function into a tool Claude can use
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
import operator
import os
import tempfile
import json

from dotenv import load_dotenv

from src.mcp_client import (
    search_web_mcp,
    search_documents_mcp
)
from src.trace_utils import extract_trace
from src.agent_utils import (
    MAX_TOOL_ROUNDS,
    has_pending_tool_calls,
    message_text,
    messages_for_final_answer,
    needs_final_answer,
)

load_dotenv()

# Global storage for uploaded files
_uploaded_files = []
_web_sources = []
_pdf_sources = []



def set_uploaded_files(files):
    global _uploaded_files
    _uploaded_files = files if files else []


class AgentState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    sources: list




@tool
def search_web(query: str) -> str:
    """Search the web through MCP and return results with source URLs."""

    global _web_sources

    raw_result = search_web_mcp(query)

    try:
        data = json.loads(raw_result)
        content = data.get("content", "")
        sources = data.get("sources", [])
        error = data.get("error")

        if error:
            return f"Web search error: {error}"

        # Save the actual URLs returned by MCP
        for url in sources:
            if url and url not in _web_sources:
                _web_sources.append(url)

        formatted_sources = "\n".join(
            f"- {url}" for url in sources
        )

        return (
            f"Search results:\n{content}\n\n"
            f"Source URLs:\n{formatted_sources}"
        )

    except (json.JSONDecodeError, AttributeError):
        return raw_result





@tool
def search_pdfs(query: str) -> str:
    """Search uploaded PDFs through MCP and return passages with citations."""

    global _uploaded_files, _pdf_sources

    if not _uploaded_files:
        return "No PDF documents have been uploaded."

    results = []

    for uploaded_file in _uploaded_files:

        # Save Streamlit UploadedFile to a temporary PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name

        try:
            raw_result = search_documents_mcp(
                query=query,
                pdf_path=pdf_path
            )

            try:
                data = json.loads(raw_result)
                content = data.get("content", "")
                sources = data.get("sources", [])
                error = data.get("error")

                if error:
                    results.append(
                        f"Document: {uploaded_file.name}\n"
                        f"Search error: {error}"
                    )
                    continue

                citations = []

                for source in sources:
                    source = str(source)

                    if ", page " in source:
                        page = source.split(", page ", 1)[1]
                        citation = f"{uploaded_file.name}, page {page}"
                    else:
                        citation = uploaded_file.name

                    if citation not in _pdf_sources:
                        _pdf_sources.append(citation)

                    if citation not in citations:
                        citations.append(citation)

                citation_text = (
                    ", ".join(citations)
                    if citations
                    else "No citations returned"
                )

                results.append(
                    f"Document: {uploaded_file.name}\n"
                    f"Relevant passages:\n{content}\n"
                    f"PDF citations: {citation_text}"
                )

            except (json.JSONDecodeError, AttributeError):
                results.append(
                    f"Document: {uploaded_file.name}\n{raw_result}"
                )

        finally:
            # Delete temporary PDF
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    return "\n\n".join(results)


def create_agent():
       
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2500,
        timeout=120,
        max_retries=2
    )


    tools = [
        search_web,
        search_pdfs
    ]

    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState):

        messages = state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "sources": state.get("sources", [])
        }

    def finalize_node(state: AgentState):
        """Last, tool-free call so the user always gets an answer."""
        response = llm.invoke(messages_for_final_answer(state["messages"]))
        return {
            "messages": [response],
            "sources": state.get("sources", [])
        }

    def should_continue(state: AgentState):

        messages = state["messages"]

        last_message = messages[-1]

        # Prevent infinite tool loops
        tool_call_count = sum(
            1
            for message in messages
            if has_pending_tool_calls(message)
        )

        if tool_call_count >= MAX_TOOL_ROUNDS:
            return "finalize" if needs_final_answer(messages) else END

        if has_pending_tool_calls(last_message):
            return "tools"

        if not message_text(last_message).strip():
            return "finalize"

        return END

    graph = StateGraph(AgentState)

    graph.add_node(
        "agent",
        agent_node
    )

    graph.add_node(
        "tools",
        tool_node
    )

    graph.add_node(
        "finalize",
        finalize_node
    )

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize",
            END: END
        }
    )

    graph.add_edge("finalize", END)

    graph.add_edge(
        "tools",
        "agent"
    )

    return graph.compile()


def orchestrator(
    query: str,
    uploaded_files=None,
    chat_history=None
) -> dict:
    global _web_sources, _pdf_sources
    _web_sources = []
    _pdf_sources = []

    set_uploaded_files(uploaded_files)

    """
    Main research orchestrator.

    The LangGraph agent uses MCP-powered tools
    for web and PDF research.
    """

    file_names = [getattr(f, "name", "document.pdf") for f in (uploaded_files or [])]
    upload_note = (
        "\n\nThe user has uploaded these PDFs, which search_pdfs can search: "
        + ", ".join(file_names) + "."
        if file_names
        else "\n\nNo PDFs are uploaded."
    )

    system_prompt = """
You are an expert research analyst.

You have access to two research tools through
the Model Context Protocol (MCP):

1. search_web
   Search the internet for current information.

2. search_pdfs
   Search uploaded PDF documents.

Research strategy:

- If PDFs are uploaded and the question could be answered from them, search the PDFs first and answer from them.
- Use search_web only for current events or outside information, when the user asks for it, or when the PDFs do not contain what is needed. If PDFs are uploaded and the question is plainly about them, do not add web research.
- If no PDFs are uploaded, use search_web for factual or current questions.
- For comprehensive research, use both tools and keep the sources separate.
- If a PDF search does not find the answer, rephrase with different key terms (names, numbers, section titles) instead of repeating the same query.
- If the answer is not in the available sources, say so plainly. Do not offer loosely related figures or facts from other sources as a substitute for the missing answer.
- Start with the answer itself. Do not comment on your own search process.
- Cross-reference information when possible.
- Never guess.
- Clearly distinguish facts from analysis.
- Cite sources whenever possible.
- Flag contradictions between sources.
- Give structured, useful answers.



Citation requirements:
- Cite only information supported by the actual search results.
- For PDF information, use the exact filename and page number returned by the PDF search tool.
- Place citations immediately after the claims they support.
- Do not cite the PDF for facts that do not appear in the retrieved passages.
- Do not invent metrics, project details, technologies, or citations.
- If the retrieved information does not support a claim, omit the claim or clearly state that the information could not be verified.
- If a response uses both web and PDF results, distinguish the sources for each claim.


"""

    system_message = SystemMessage(content=system_prompt + upload_note)

    messages = [system_message]

    # Add recent conversation history
    if chat_history:

        for msg in chat_history[-6:]:
            messages.append(msg)

    messages.append(
        HumanMessage(content=query)
    )

    agent = create_agent()

    initial_state = {
        "messages": messages,
        "sources": []
    }

    result = agent.invoke(initial_state)

    # Get final answer
    final_message = result["messages"][-1]

    answer = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )


    # Extra fields used by the eval harness (eval/run_eval.py); the app ignores them.
    tool_calls, contexts = extract_trace(result["messages"])

    return {
        "answer": answer,
        "sources": _web_sources + _pdf_sources,
        "web_sources": _web_sources.copy(),
        "pdf_sources": _pdf_sources.copy(),
        "query": query,
        "tool_calls": tool_calls,
        "contexts": contexts
    }
