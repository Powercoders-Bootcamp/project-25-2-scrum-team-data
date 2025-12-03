# langgraph_app/state.py

from typing import Any, Dict, List
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ChatState(TypedDict):
    """
    State that flows through the LangGraph.

    - messages: full chat history as LangChain messages
                (HumanMessage / AIMessage / SystemMessage).
      `Annotated[..., add_messages]` means new messages returned by
      nodes will be *appended* instead of replacing the list.

    - last_retrieved: list of retrieved chunks for the latest user turn,
      each as {"metadata": {...}, "snippet": "..."}.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    last_retrieved: List[Dict[str, Any]]
