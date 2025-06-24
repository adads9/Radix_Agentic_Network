import os, sys, asyncio
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
import functools
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.azure_openai import (
    azure_openai_embed,
    azure_openai_complete_if_cache,
)
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.tools import BaseTool
from openevals.llm import create_llm_as_judge          
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
)                                                   

load_dotenv()

#  Environment variables for the agent
docs_agent_model      = os.getenv("DOCS_AGENT_LLM_MODEL")
embed_model           = os.getenv("RETRIEVER_EMBED_MODEL")
retrieve_model        = os.getenv("RETRIEVER_LLM_MODEL")
api_key               = os.getenv("API_KEY")
api_version           = os.getenv("BASE_LLM_API_VERSION")
api_embed_version     = os.getenv("EMBED_API_VERSION")
azure_endpoint        = os.getenv("BASE_API_BASE")
azure_embed_endpoint  = os.getenv("EMBED_API_BASE")
azure_point           = os.getenv("BASE_FOR_LIGHTRAG")
lightrag_mode         = os.getenv("LIGHTRAG_MODE")
judge_llm             = os.getenv("JUDGE_LLM_MODEL")
judge                 = os.getenv("JUDGE_RESPONSE")



# ────────────────────────────────────────────────────────────────────────
# ───────────── 1.  Embedding setup and LightRAG Initialization ──────────
# ────────────────────────────────────────────────────────────────────────

if embed_model == "text-embedding-3-large":
    WORKING_DIR, embedding_dim, max_token_size = "./radix_docs_embed_3072", 3072, 8191
else:
    WORKING_DIR, embedding_dim, max_token_size = "./radix_docs", 1536, 8191

os.makedirs(WORKING_DIR, exist_ok=True)

async def embedding_func(texts: list[str]):
    return await azure_openai_embed(
        model=embed_model,
        texts=texts,
        api_key=api_key,
        base_url=azure_point,
        api_version=api_embed_version,
    )

async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    return await azure_openai_complete_if_cache(
        retrieve_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=azure_point,
        api_version=api_version,
        **kwargs
    )

# Initialize LightRAG with the embedding function and LLM model function
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=embedding_func,
        ),
        llm_model_func=llm_model_func,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

_global_rag = asyncio.run(initialize_rag())


# ────────────────────────────────────────────────────────────────────────
# ─────────────────── 2.  LLM-as-Judge plumbing ──────────────────────────
# ────────────────────────────────────────────────────────────────────────


def _build_judge_llm() -> AzureChatOpenAI:
    """
    Deterministic model for scoring.
    Temperature = 0 keeps judgments stable across runs.
    """
    return AzureChatOpenAI(
        model=judge_llm,
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_point,
        temperature=0.0,
        streaming=False,
    )                                                  

# Two evaluators from OpenEvals (RAG triad: relevance + groundedness)
_judge_llm = _build_judge_llm()

retrieval_relevance_eval = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    judge=_judge_llm,
)                                                     

groundedness_eval = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    judge=_judge_llm,
)                                                     

class RadixJudge:
    """Bundle both scores + pass/fail flag."""
    def __init__(self, retrieval_eval, grounded_eval):
        self._ret_eval = retrieval_eval
        self._grd_eval = grounded_eval

    def __call__(self, *, question: str, answer: str, context):
        ret_score = self._ret_eval(inputs=question, context=context)
        grd_score = self._grd_eval(outputs=answer, context=context)
        return {
            "retrieval_relevance": ret_score,
            "groundedness": grd_score,
            "overall_pass": bool(ret_score["score"] and grd_score["score"]),
        }

llm_judge = RadixJudge(retrieval_relevance_eval, groundedness_eval)

# ────────────────────────────────────────────────────────────────────────
# ─────────────────── 3. Tools for the Docs Agent  ───────────────────────
# ────────────────────────────────────────────────────────────────────────

@tool
async def retrieve(search_query: str) -> str:
    """
    Search Radix docs and return the context needed to answer the query"
      output: 'contexts': [str] of retrieved passages
    """
    result = await _global_rag.aquery(
        search_query,
        param=QueryParam(
            mode=lightrag_mode,   # we want answer + context  # **NEW** as of LightRAG ≥0.5.0
        )
    )
    return result  # {"answer": str, "contexts": [...]}  :contentReference[oaicite:6]{index=6}


@tool
async def judge_response(response: str, question: str, context: list[str]) -> dict:
    """
    Judge whether `response` fully and faithfully answers `question`
    given LightRAG `context` chunks.

    inputs:
    - response (generated answer)
    - question (original query from user)
    - context (retrieved context from retriever)


    """
    

    loop = asyncio.get_running_loop()
    fn = functools.partial(
        llm_judge,
        question=question,
        answer=response,
        context=context,
    )
    return await loop.run_in_executor(None, fn)

# ────────────────────────────────────────────────────────────────────────
# 3.  Experimental activation (prompts, tools)
# ────────────────────────────────────────────────────────────────────────

prompt_for_retrival=("You are a knowledgeable assistant designed to answer questions about Radix documentation.\n"
                    f"Always Use the 'retrieve' tool to extract pertinent information from the documentation.\n"
                    "If the information is unavailable, provide your best general knowledge response. "
                    "If you do not know the answer to the question, or cannot find the information in the docs, do not make up an answer and inform the user appropiatly.\n"
                    
                    "RULES:\n"
                    "Make sure to always use the most relevant and up-to-date information from the Radix documentation.\n"
                    "Whenever you are asked about Radix or any documentation, know that you are supposed to use the 'retrieve' tool to get the most accurate and relevant information.\n"
                    "If the question is related to radix config, either to modify or to create a new one, please ensure that you are using the correct syntax and format.\n"
                    "You should also do a deep-dive in the specifics of the radix config documentation, ensuring that the produced result is including every detail that it needs.\n"
                    "If the information is unavailable, provide your best general knowledge response. "
                    "If you do not know the answer to the question, or cannot find the information in the docs, do not make up an answer and inform the user appropriately.\n")

prompt_for_retrival_judge=("You are a knowledgeable assistant designed to answer questions about Radix documentation.\n"
                            f"Always use the 'retrieve' tool to extract pertinent information from the documentation, "
                            "and generate an answer for the question based on the information from the documentation.\n"
                            "After the answer is generated, "
                            "you should always use the 'judge_response' tool to evaluate the generated answer and the retrieved information.\n" 
                            "Only give the FINAL answer back when it has passed the 'judge_respone' tool, and the 'overall_pass' is 'true'\n\n"
                            "You should use the feedback from the 'judge_response' to search for the reason it has failed, and make sure to fix it with the correct information."
                            
                            "RULES:\n"
                            "Make sure to always use the most relevant and up-to-date information from the Radix documentation.\n"
                            "Whenever you are asked about Radix or any documentation, know that you are supposed to use the 'retrieve' tool to get the most accurate and relevant information.\n"
                            "If the question is related to radix config, either to modify or to create a new one, please ensure that you are using the correct syntax and format.\n"
                            "You should also do a deep-dive in the specifics of the radix config documentation, ensuring that the produced result is including every detail that it needs.\n"
                            "If the information is unavailable, provide your best general knowledge response. "
                            "If you do not know the answer to the question, or cannot find the information in the docs, do not make up an answer and inform the user appropriately.\n")



tools=[retrieve]
prompt_to_use=prompt_for_retrival

# If judge is enabled, add the judge_response tool and change the prompt with the judge prompt. 
# judge is a value that can be turned on or off in the .env file, and is used to determine if the judge_response tool should be used or not.

if judge=='on':
    tools.append(judge_response)
    prompt_to_use=prompt_for_retrival_judge
    print("Judge response is turned on, using the judge_response tool for the docs agent. To turn it off, set JUDGE_RESPONSE=off in the .env file.")
elif judge=='off':
    print("Judge response is turned off, not using the judge_response tool for the docs agent. To turn it on, set JUDGE_RESPONSE=on in the .env file.")
else:
    print("Judge_response tool is not set, or is invalid, please set JUDGE_RESPONSE=on or JUDGE_RESPONSE=off in the .env file.")


# ────────────────────────────────────────────────────────────────────────
# ────────────────────────── 4.  Agent setup ─────────────────────────────
# ────────────────────────────────────────────────────────────────────────

# Create the LLM for the docs agent
llm = AzureChatOpenAI(                                
    model=docs_agent_model,
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    streaming=True,
)

# Create the React agent for the docs
docs_agent = create_react_agent(
    model=llm,
    tools=tools,
    name="docs_agent",
    prompt=prompt_to_use,
)
