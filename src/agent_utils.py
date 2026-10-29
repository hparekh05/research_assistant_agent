"""Small helpers for the agent loop. Kept separate so they can be tested without the API."""
from langchain_core.messages import AIMessage, HumanMessage

MAX_TOOL_ROUNDS = 4

FINALIZE_PROMPT = (
    "You have reached the search limit. Answer the question now using only the information "
    "retrieved above. Cite the file name and page for PDF facts. If the retrieved information "
    "does not contain the answer, say clearly that it is not stated in the available sources. "
    "Do not request more searches."
)


def message_text(msg) -> str:
    """Claude sometimes returns a list of content blocks instead of a plain string."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def has_pending_tool_calls(msg) -> bool:
    return bool(getattr(msg, "tool_calls", None))


def needs_final_answer(messages) -> bool:
    """True when the run is about to end without usable text (unanswered tool request or empty reply)."""
    last = messages[-1]
    return has_pending_tool_calls(last) or not message_text(last).strip()


def messages_for_final_answer(messages):
    """History to send for a last, tool-free answer.

    A trailing AIMessage that asked for tools but never got results is dropped, because the API
    rejects a tool request that has no matching result.
    """
    msgs = list(messages)
    if msgs and (has_pending_tool_calls(msgs[-1])
                 or (isinstance(msgs[-1], AIMessage) and not message_text(msgs[-1]).strip())):
        msgs = msgs[:-1]
    msgs.append(HumanMessage(content=FINALIZE_PROMPT))
    return msgs
