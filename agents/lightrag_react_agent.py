import os
import asyncio
import sys
from pathlib import Path

# Add the project root to sys.path:
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import numpy as np
from openai import AsyncAzureOpenAI, AsyncOpenAI
from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from lightrag.utils import EmbeddingFunc
from openevals.llm import create_llm_as_judge

load_dotenv()

embed_model = os.getenv("RETRIEVER_EMBED_MODEL")
llm_model=os.getenv("RETRIEVER_LLM_MODEL")  # e.g. gpt-4o-mini
api_key=os.getenv("API_KEY")
api_version=os.getenv("BASE_LLM_API_VERSION")
api_embed_version=os.getenv("EMBED_API_VERSION")
azure_endpoint=os.getenv("BASE_API_BASE")
azure_embed_endpoint=os.getenv("EMBED_API_BASE")
azure_point=os.getenv("BASE_FOR_LIGHTRAG")
lightrag_mode=os.getenv("LIGHTRAG_MODE")

print(llm_model)
# Specify the working directory based on the correct embedding model.

if embed_model == "text-embedding-3-large":
    WORKING_DIR = "./radix_docs_embed_3072"
    embedding_dim = 3072
    max_token_size = 8191
elif embed_model == "text-embedding-3-small":
    embedding_dim = 1536
    max_token_size = 8191
    WORKING_DIR = "./radix_docs"
else:
    raise ValueError(f"Unsupported embed_model: {embed_model}. Expected 'text-embedding-3-large' or 'text-embedding-3-small'.")
print("Working directory:", WORKING_DIR)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt,
    system_prompt= None,
    history_messages= [],
    keyword_extraction = False,
    **kwargs
) -> str:
    """
    Completion function for the LightRAG Agent.

    This function is called by the LightRAG Agent to generate completions for the user's query.
    It uses the Azure OpenAI API to call the specified deployment (llm_model) and return the generated text.

    Args:
        prompt (str): The input prompt for the completion.
        system_prompt (str, optional): The system prompt for the completion. Defaults to None.
        history_messages (list, optional): The history messages for the completion. Defaults to [].
        keyword_extraction (bool, optional): Whether to perform keyword extraction. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the Azure OpenAI API.

    Returns:
        str: The generated completion text.
    """
    return await azure_openai_complete_if_cache(
        llm_model,   # deployment *name*
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key = api_key,
        base_url = azure_point,
        api_version=api_version,
        **kwargs
    )

"""
_azure = AsyncAzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,
)

print("Azure client initialized:", _azure)

async def llm_model_func(prompt, system_prompt=None, history_messages=None,
                         keyword_extraction=False, **kwargs) -> str:
    # LightRAG ≤0.3 passes a str; ≥0.4 passes list[dict]
    # 1. Normalize prompt → list[dict]
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, dict):
        messages = [prompt]
    elif isinstance(prompt, list):
        messages = []
        for item in prompt:
            if hasattr(item, "model_dump"):
                messages.append(item.model_dump())
            elif isinstance(item, dict):
                messages.append(item)
            else:
                messages.append({"role":"user","content": str(item)})
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    # Azure client – no extra 'model' param! use deployment in URL instead
    resp = await _azure.chat.completions.create(
        model=llm_model,  # deployment *name*
        messages=messages,
        stream=kwargs.get("stream", False),  # pass-through if present
    )
    return resp.choices[0].message.content


_azure_embeddings = AsyncAzureOpenAI( 
    api_key=api_key,
    api_version=api_embed_version,
    azure_endpoint=azure_embed_endpoint,
) """
async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    Asynchronously obtains embeddings for a list of texts using Azure OpenAI.

    Args:
        texts (list[str]): A list of strings for which embeddings are to be generated.

    Returns:
        np.ndarray: An array of embeddings corresponding to the input texts.
    """

    return await azure_openai_embed(
        model=embed_model,  # e.g. text-embed-3-sm
        texts=texts,
        api_key=api_key,
        base_url=azure_point,
        api_version=api_embed_version
    )  

# Initialize the LightRAG instance asynchronously—instead of the extra pydantic wrapper.
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=max_token_size, func=embedding_func),            # 👈 custom function
        llm_model_func=llm_model_func,   # or your Azure chat deployment
    )
    await rag.initialize_storages()
    return rag

_global_rag = asyncio.run(initialize_rag())



@tool
async def retrieve(search_query: str) -> str:
    """Search Radix docs and return a synthesis.

    mode: Specifies the retrieval mode:

        - "local": Focuses on context-dependent information.
        - "global": Utilizes global knowledge.
        - "hybrid": Combines local and global retrieval methods.
        - "naive": Performs a basic search without advanced techniques.
        - "mix": Integrates knowledge graph and vector retrieval.
    """
    return await _global_rag.aquery(search_query, param=QueryParam(mode=lightrag_mode))


# Create the LLM judge for the agent.
llm_judge = create_llm_as_judge(
    model=llm_model_func,
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# judge the agent's responses.
@tool
async def judge_response(response: str, question: str, context: str) -> str:
    return await llm_judge.judge_response(response, question, context)



# Create the main LLM instance with streaming enabled.
llm = AzureChatOpenAI(
    model=llm_model,
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint, 
)    

# Create the docs_agent as a react agent.
docs_agent = create_react_agent(
    model=llm,
    tools=[retrieve],
    name="docs_agent",
    prompt=(
        "You are a knowledgeable assistant designed to answer questions about Radix documentation.\n"
        "Use the retrieve tool to extract pertinent information from the documentation.\n"
        "Make sure to always use the most relevant and up-to-date information from the Radix documentation.\n"
        "Whenever you are asked about Radix or any documentation, know that you are supposed to use the retrieve tool to get the most accurate and relevant information.\n"
        "If the question is related to radix config, either to modify or to create a new one, please ensure that you are using the correct syntax and format.\n"
        "You should also do a deep-dive in the specifics of the radix config documentation, ensuring that the produced result is including every detail that it needs.\n"
        "If the information is unavailable, provide your best general knowledge response. "
        "If you do not know the answer to the question, or cannot find the information in the docs, do not make up an answer and inform the user appropriately.\n"
    )
)