# langgraph_app/nodes.py

"""
LangGraph nodes for the RAG chat agent.

We use a single `agent_node` that:
  1. Looks at the latest user message in the state.
  2. Retrieves + reranks documents from Chroma.
  3. Builds a RAG prompt and calls the HF LLM.
  4. Appends the assistant answer and stores retrieved chunks.
"""

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from vector_pipeline.retrieval import retrieve_documents
from .state import ChatState
from .config import get_vectorstore, get_llm


def _get_last_user_message(messages: List[BaseMessage]) -> HumanMessage:
    """
    Return the last HumanMessage in the list.

    We scan backwards to be safe; if none is found, we error.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    raise ValueError("No HumanMessage found in state.messages.")


def agent_node(state: ChatState) -> ChatState:
    """
    Core RAG agent node.

    Reads:
      - state["messages"]  (full history)

    Writes:
      - appends an AIMessage with the answer to `messages`
      - overwrites `last_retrieved` with the chunks for this turn
    """
    vs = get_vectorstore()
    llm = get_llm()

    last_user = _get_last_user_message(state["messages"])
    query = last_user.content

    # 1) Retrieve docs with your existing function (reranker on)
    docs = retrieve_documents(
        query=query,
        vs=vs,
        k=4,
        use_reranker=True,
    )

    if not docs:
        answer_text = "I couldn't find anything relevant in the product database."
        retrieved: List[Dict[str, Any]] = []
    else:
        # Prepare retrieved chunks for UI + context
        retrieved = [
            {
                "metadata": doc.metadata,
                "snippet": doc.page_content[:400],
            }
            for doc in docs
        ]

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Use ONLY the following product information to answer the question.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()

        # HuggingFacePipeline.invoke usually returns a string
        answer_text = llm.invoke(prompt)

    ai_msg = AIMessage(content=answer_text)

    # Because of `add_messages`, returning `{"messages": [ai_msg]}`
    # tells LangGraph: "append this to the existing list".
    return {
        "messages": [ai_msg],
        "last_retrieved": retrieved,
    }
