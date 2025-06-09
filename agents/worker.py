# workers.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI          #LangChain wrapper
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from github_agent import github_agent, run_with_retry  # import the retry wrapper
# Bridge to the MCP-backed GitHub agent to execute GitHub actions. Needed due to
# the MCP server being run in a separate thread.

load_dotenv()


@tool
async def github_exec(instruction: str) -> str:
    """
    Execute *any* GitHub instruction (list repos, search code, commit files, open pull requests, etc.)
    using the GitHub MCP agent. This function is a bridge to the MCP server.
    """
    result = await run_with_retry(github_agent, instruction)   # no nested event loop
    return result.data

llm = AzureChatOpenAI(
    model=os.getenv("BASE_LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("BASE_LLM_API_VERSION"),
    base_url=os.getenv("BASE_API_BASE"), 
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()]
)    


# github agent made into a react agent for the supervisor
github_react_agent = create_react_agent(
    model=llm,
    tools=[github_exec],
    name="github_agent",
    prompt=(
        "You are a GitHub automation agent connected through MCP.\n"
        "Use the github_exec tool to perform *any* action (list repos, "
        "search code, commit files, open pull requests, etc.).\n"
        "Think step-by-step; call the tool as many times as needed until "
        "the task is complete, then reply with FINAL."
        "If there is an error, retry or state that an error occurred and suggest a fallback.\n"
    ),
)
