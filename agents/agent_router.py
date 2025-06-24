import asyncio
#asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) # for Windows compatibility (Langgraph Studio)
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from github_agent import github_agent # use (agents.github_agent) when using langgraph studio
from langchain_core.messages import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from lightrag_react_agent import docs_agent # only use (agents.lightrag_react_agent) when using langgraph studio
import os
from radix_validate import radix_linter_agent # only use (agents.radix_validate) when using langgraph studio
import os
from dotenv import load_dotenv
load_dotenv()


llm = AzureChatOpenAI(
    model=os.getenv("SUPERVISOR_LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("BASE_LLM_API_VERSION"), 
    azure_endpoint=os.getenv("BASE_FOR_LIGHTRAG"),
    streaming=True,
    #callbacks=[StreamingStdOutCallbackHandler()],  # Uncomment for streaming output (streaming is tricky with several agents, so use with care)
)    

supervisor_graph = (
    create_supervisor(
        agents=[docs_agent, github_agent, radix_linter_agent],     # three workers now
        model=llm,
        prompt = """
                You are the *Supervisor* in a multi-agent LangGraph system.


                ────────────────────  TEAM ROSTER  ────────────────────
                • docs_agent  
                  - Agent looks up Radix documentation and answer all questions regarding radix. It can generate complete artifacts, f.ex radixconfig.yaml, wrapped in fenced code blocks.
                  - Responsible for drafting files such as the radixconfig.yaml when requested.

                • github_agent  
                  - Uses tools from the Github MCP server that for ANY GitHub-related instruction (e.g. list repos, query user info, commit files, open PRs, etc.).

                • radix_linter_agent  
                  - Validates a draft `radixconfig.yaml` using validator tools to report schema and style violations.
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
                   - If a user asks for modifications on their radixconfig file or requests its creation, first instruct **docs_agent** to get the general info of all the fields in a radixconfig, or for creation to generate a draft version based on radix config docs.
                   - Then, delegate to **radix_linter_agent** to validate and apply minimal patches so the file adheres to the radix validation rules.
                   - Only after validation is complete should the final version be passed back to the user.

                ────────────────────  BEHAVIOUR RULES  ────────────────────
                • Never call tools yourself — always delegate.
                • When delegating, please ensure that you give the agent only the part of the task that it is supposed to do, as additional information might confuse the agent.
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
    .compile(checkpointer=MemorySaver(), name="Supervisor") # add checkpointer=MemorySaver() as a parameter to save the conversation history (Remove when using langgraph studio)
)

async def chat():
    thread_id = "cli"
    print("Supervisor ready. Type 'exit' to quit.")
    # supervisor start-up with the MCP servers for the github agent
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
