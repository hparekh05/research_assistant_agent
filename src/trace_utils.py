"""Helpers that pull tool-call details out of a LangGraph message list.

Used by the orchestrator so the eval harness can see which tools ran, whether they worked,
and what text the answer was built from. Importing this has no effect on the app itself.
"""
from langchain_core.messages import ToolMessage

# The tools return error text instead of raising, so a "successful" ToolMessage can still be a failure.
_FAILURE_MARKERS = (
    "web search error",
    "web search unavailable",
    "web search failed",
    "search error:",
    "pdf search failed",
    "pdf search encountered an error",
    "no pdf documents have been uploaded",
    "no results returned from mcp",
    "no document results returned from mcp",
)


def _text(content) -> str:
    """Claude sometimes returns a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def tool_succeeded(content) -> bool:
    low = _text(content).lower()
    return not any(marker in low for marker in _FAILURE_MARKERS)


def extract_trace(messages):
    """Return (tool_calls, contexts) from a LangGraph run.

    tool_calls: [{"name": "search_web", "ok": True}, ...] in call order
    contexts:   the raw text each tool returned (what the answer was grounded in)
    """
    tool_calls, contexts = [], []
    for m in messages:
        if isinstance(m, ToolMessage):
            text = _text(m.content)
            ok = getattr(m, "status", "success") != "error" and tool_succeeded(text)
            tool_calls.append({"name": m.name or "unknown", "ok": ok})
            contexts.append(text)
    return tool_calls, contexts
