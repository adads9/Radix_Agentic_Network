from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from worker import github_react_agent
from langchain_core.messages import HumanMessage
import asyncio
from github_agent import github_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from lightrag_react_agent import docs_agent
import subprocess, atexit, pathlib
import os

import os, subprocess, atexit, pathlib, sys
from dotenv import load_dotenv
load_dotenv()

SPECTRAL_IMAGE = "stoplight/spectral:latest"
CONTAINER_NAME = "spectral-linter"


# Pull the Spectral image at startup
subprocess.run(["docker", "pull", SPECTRAL_IMAGE], check=True)


from radix_validate import radix_linter_agent

llm = AzureChatOpenAI(
    model=os.getenv("SUPERVISOR_LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("BASE_LLM_API_VERSION"), 
    base_url=os.getenv("BASE_API_BASE"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],  # Uncomment for streaming output 
)    

supervisor_graph = (
    create_supervisor(
        agents=[docs_agent, github_react_agent, radix_linter_agent],     # three workers now
        model=llm,
        prompt = """
                You are the *Supervisor* in a multi-agent LangGraph system.

                ────────────────────  TEAM ROSTER  ────────────────────
                • docs_agent  
                  - Uses `retrieve` to look up Radix documentation and answer all questions regarding radix. It can generate complete artifacts, f.ex radixconfig.yaml, wrapped in fenced code blocks.
                  - Responsible for drafting files such as the radixconfig.yaml when requested.

                • github_agent  
                  - Uses a single tool, `github_exec`, that forwards ANY GitHub-related instruction (e.g. list repos, query user info, commit files, open PRs, etc.) to the GitHub MCP server.

                • radix_linter_agent  
                  - Validates a draft `radixconfig.yaml` using deterministic validators with the tools validate_yaml & apply_patch to report schema and style violations.
                  - Automatically applies patches when possible so the file passes `radix validate`.

                ────────────────────  ROUTING POLICY  ────────────────────
                1. **Docs-only information** (answers that can be satisfied from Radix docs without touching other agents) → delegate to **docs_agent**.
                
                2. **GitHub-dependent information** (requires live data from the user's GitHub account, e.g. “What is my GitHub username?”, “How many open PRs do I have?”) → delegate to **github_agent**, even if it’s purely informational.
                
                3. **Pure GitHub actions** (commit, list, PR, etc.) → delegate to **github_agent**.
                
                4. **Combined requests** (needs docs **and** GitHub), e.g. “Summarise Radix in three sentences and update my README”:
                   a. transfer_to_docs_agent - get the summary or file fragment.
                   b. Wait for a fenced code block.
                   c. transfer_to_github_agent - commit/PR.
                   d. Reply FINAL only after the PR URL is available.
                
                5. **Radix Configuration Tasks**:
                   - If a user asks for modifications on their radixconfig file or requests its creation, first instruct **docs_agent** to generate a draft version.
                   - Then, delegate to **radix_linter_agent** to validate and apply minimal patches so the file adheres to the radix validation rules.
                   - Only after validation is complete should the final version be passed back to the user.

                ────────────────────  BEHAVIOUR RULES  ────────────────────
                • Never call tools yourself — always delegate.
                • Handle errors politely (e.g., missing docs → apologise; GitHub auth failure → apologise & ask for next steps).
                • When you feel you have finished a task, check whether the user needs further actions (e.g., additional modifications using the github_agent after the docs_agent).

                ────────────────────  OUTPUT RULES  ────────────────────
                • FINAL replies must include (when applicable):
                  - Answers or summaries requested,
                  - File path and repository,
                  - Pull-request URL (if applicable).

                Begin supervising.
                """,
        add_handoff_back_messages=True,
        output_mode="full_history",
    )
    .compile(checkpointer=MemorySaver())
)

async def chat():
    thread_id = "cli"
    print("Supervisor ready. Type 'exit' to quit.")
    # supervisor start-up with the MCP servers for the github agent
    async with github_agent.run_mcp_servers():
        while True:
            user = input("\nYou: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            result = await supervisor_graph.ainvoke( 
                {"messages": [HumanMessage(content=user)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            print("\nAssistant:", result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(chat())


""" Work needed with the radix linter agent and the ruleset, it is working now but
    needs to be tested with the ruleset and the apply patch tool. Maybe look at a mcp for this instead of the docker container.
    """