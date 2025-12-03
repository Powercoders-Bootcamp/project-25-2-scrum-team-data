# langgraph_app/graph.py

"""
LangGraph app for your product RAG chatbot.

Flow:
    START -> agent_node -> END

We compile it with an InMemorySaver checkpointer, so history is stored
per `thread_id` that the backend passes in the config.
"""

from typing import Any, Dict, List

from typing_extensions import Literal

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ChatState
from .nodes import agent_node


# ----------------- Build & cache the graph --------------------


_app = None
_memory = MemorySaver()  # per-thread chat history


def get_app():
    """
    Return a compiled LangGraph app with memory.
    """
    global _app
    if _app is not None:
        return _app

    workflow = StateGraph(ChatState)

    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    _app = workflow.compile(checkpointer=_memory)
    return _app


# -------------- Helpers: dict <-> LC messages ------------------


def _dict_to_lc_message(msg: Dict[str, Any]) -> BaseMessage:
    role: str = msg.get("role", "user")
    content = msg.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)

    # unknown role → treat as user
    return HumanMessage(content=str(content))


def _lc_message_to_dict(msg: BaseMessage) -> Dict[str, Any]:
    if isinstance(msg, HumanMessage):
        role: Literal["user", "assistant", "system"] = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, SystemMessage):
        role = "system"
    else:
        role = "assistant"

    return {"role": role, "content": msg.content}


# ---------------- Public entrypoints for backend ----------------


def run_chat_session(
    user_message: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    **Stateful** chat using LangGraph memory.

    Backend passes:
      - user_message: latest user text
      - session_id: identifier for this conversation
                    (IP, cookie, auth user id, etc.)

    LangGraph:
      - loads previous state for this session (thread_id)
      - appends the new HumanMessage
      - runs the graph
      - stores updated state back under the same thread_id

    Returns:
        {
          "answer": str,
          "messages": list[{"role","content"}],  # full history
          "retrieved": list[{"metadata","snippet"}]
        }
    """
    app = get_app()

    # New input for this turn
    state_in: Dict[str, Any] = {
        "messages": [HumanMessage(content=user_message)]
        # last_retrieved will be set by the node
    }

    config = {"configurable": {"thread_id": session_id}}

    state_out: ChatState = app.invoke(state_in, config=config)

    lc_messages = state_out["messages"]
    retrieved = state_out.get("last_retrieved", [])

    messages = [_lc_message_to_dict(m) for m in lc_messages]
    answer = messages[-1]["content"] if messages else ""

    return {
        "answer": answer,
        "messages": messages,
        "retrieved": retrieved,
    }


def run_chat_stateless(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    **Stateless** variant: backend sends full history each time.

    - No thread_id / memory used.
    - Useful for testing or if the backend prefers to store history itself.
    """
    app = get_app()

    lc_messages = [_dict_to_lc_message(m) for m in messages]
    state_in: Dict[str, Any] = {"messages": lc_messages}

    state_out: ChatState = app.invoke(state_in)

    lc_messages_out = state_out["messages"]
    retrieved = state_out.get("last_retrieved", [])

    messages_out = [_lc_message_to_dict(m) for m in lc_messages_out]
    answer = messages_out[-1]["content"] if messages_out else ""

    return {
        "answer": answer,
        "messages": messages_out,
        "retrieved": retrieved,
    }
