import os
import re
import markdown
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .ingestion import build_or_load_vectorstore
from .config import get_bge_reranker
from .retrieval import retrieve_documents
from .settings import (CHROMA_DIR, 
                       OPENROUTER_API_KEY_PATH, 
                       CLOUD_LLM_MODEL_NAME, 
                       EMBEDDING_MODEL_NAME)



# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def get_prompt_template() -> PromptTemplate:
    """
    Return the prompt template for RAG QA.
    """

    prompt_template = """You are a product knowledge assistant specialized in musical instruments and music-related equipment.
Your job is to answer user questions using ONLY the information contained in the retrieved context documents.

The dataset you are working with contains:
- product title, description, features, categories
- product metadata (store, color, rating, rating count, price, etc.)

### Your Rules:
1. Use ONLY the retrieved context to answer questions.  
2. Do NOT invent product details, specifications, or metadata that are not present in the context.  
3. If the context does not contain the required information, say:  
   "The provided product information does not include this detail."
4. If the question is about comparing products, create a clear comparison using only the available data.
5. Summaries must be concise and factual.
6. Preserve any numerical values exactly as they appear in the context.
7. If the user asks about availability or stock, respond with:  
   "This dataset does not include real-time availability information."
8. When the question is unclear, ask for clarification.
9. If metadata is available in the context (e.g., store, color, rating), include it in your answer.
10. NEVER output raw JSON or database structures—respond in clean natural language.

### Response Format:
- Always provide a short, direct answer first.
- If relevant, include a structured breakdown:
  - **Key Features**
  - **Specifications / Metadata**
  - **Summary**

### Context:
{context}

### User Question:
{question}

### Final Answer:
(Your answer here)
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    print("Prompt was created successfully.")

    return PROMPT

# ---------------------------------------------------------------------------
# LLM instance
# ---------------------------------------------------------------------------
def get_llm() -> ChatOpenAI:
    """
    Return a ChatOpenAI LLM instance using OpenRouter.

    The OPENROUTER_API_KEY is read from the .env file at
    OPENROUTER_API_KEY_PATH.
    """

    # Load API key from .env file
    load_dotenv(dotenv_path=OPENROUTER_API_KEY_PATH, override=True)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if OPENROUTER_API_KEY:
        print("✅ API key was loaded successfully!")
        print(f"📍 .env file location: {OPENROUTER_API_KEY_PATH}")
    else:
        print("❌ API key couldn't be found. Please check the .env file.")
        print(f"🔍 Searched location: {OPENROUTER_API_KEY_PATH}")

    # Create LLM instance with OpenRouter headers
    llm = ChatOpenAI(
        model_name=CLOUD_LLM_MODEL_NAME[0],
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        temperature=0.7,
        max_tokens=2000,
    )

    print("LLM instance created successfully.")

    return llm


# ---------------------------------------------------------------------------
# Format retrieved documents 
# ---------------------------------------------------------------------------
def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(
            f"### Document {i+1}\n"
            f"Content:\n{doc.page_content}\n\n"
            f"Metadata:\n{doc.metadata}\n"
        )
    return "\n\n".join(formatted)

# ---------------------------------------------------------------------------
# RAG pipeline function
# ---------------------------------------------------------------------------
def ask_question(
    question: str,
    k: int = 10,
    use_reranker: bool = True,
) -> str:
    """
    Run the full RAG pipeline:
    1) Retrieve top-k relevant documents from Chroma DB.
    2) Format them and create the prompt.
    3) Call the LLM to generate the answer.

    Parameters
    ----------
    question : str
        The user question to answer.
    k : int, optional
        Number of documents to retrieve, by default 10
    use_reranker : bool, optional
        Whether to use the cross-encoder reranker, by default True

    Returns
    -------
    answer : str
        The generated answer from the LLM.
    """

    # 1) Load or build vectorstore
    vs: Chroma = build_or_load_vectorstore()

    # 2) Retrieve documents
    docs: List[Document] = retrieve_documents(
        query=question,
        vs=vs,
        k=k,
        use_reranker=use_reranker,
    )

    # 3) Format retrieved documents
    context = format_docs(docs)

    # 4) Create prompt
    prompt = get_prompt_template()

    # 5) Get LLM instance
    llm = get_llm()

    # 6) Generate answer
    response = llm.invoke(prompt.format(context=context, question=question))
    answer = getattr(response, "content", "")
    
    # 7) Convert answer to HTML
    html_answer = convert_answer_to_html(answer)
    return html_answer

# ---------------------------------------------------------------------------
# Convert answer to HTML
# ---------------------------------------------------------------------------

def convert_answer_to_html(answer: str) -> str:
    """
    Convert the LLM answer to HTML format for better display.

    Parameters
    ----------
    answer : str
        The raw answer from the LLM.

    Returns
    -------
    html_answer : str
        The answer converted to HTML format.
    """

    modified_answer = re.sub(r"(\*\*.+?\*\*)\n\*", r"\1\n\n*", answer)
    html_answer = markdown.markdown(modified_answer)
    return html_answer