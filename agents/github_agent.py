from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from langgraph.graph.message import Messages

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from langgraph.graph.message import add_messages
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncAzureOpenAI           # new OpenAI v1.x client
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider

load_dotenv()

azure_client= AsyncAzureOpenAI(azure_endpoint=os.getenv("BASE_API_BASE"), api_key=os.getenv("API_KEY"), api_version=os.getenv("BASE_LLM_API_VERSION"))
llm = os.getenv("BASE_LLM_MODEL")
model = OpenAIModel(model_name=llm, provider=AzureProvider(openai_client=azure_client))

logfire.configure(send_to_logfire="if-token-present")

# Prompt user for token (visible input here; in real usage consider getpass fallback if possible)
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

system_prompt = """
You are the GitHub Agent, connected to GitHub via MCP. Your primary responsibility is to interact with GitHub repositories and perform various GitHub operations accurately.

AVAILABLE TOOLS AND WHEN TO USE THEM:

1. Repository Operations:
   - Use bb7_get_me to verify authentication and get user details
   - Use bb7_search_repositories to find repositories (including private ones)
   - Use bb7_list_branches to list repository branches

2. File Operations:
   - Use bb7_get_file_contents to read any file (including README.md). IMPORTANT: The content will be base64 encoded - you must decode it first!
   - Use bb7_create_or_update_file to update a single file. IMPORTANT: Do not send base64 encoded content, send plain text!
   - Use bb7_push_files for multi-file updates. IMPORTANT: Send plain text content, not base64 encoded!

3. Issue & PR Management:
   - Use bb7_list_issues or bb7_search_issues to find issues
   - Use bb7_create_issue for new issues
   - Use bb7_create_pull_request for new PRs
   - Use bb7_get_pull_request to check PR details

4. Error Handling Protocol:
   - If an operation fails, ALWAYS verify authentication first with bb7_get_me
   - For file operations, verify repository access before claiming no access
   - When updating files, always fetch current content first
   - If a tool fails, try an alternative tool before giving up

IMPORTANT RULES:
1. ALWAYS verify authentication before claiming lack of access
2. For file updates, ALWAYS get current content first and decode from base64
3. When updating files, ALWAYS send plain text (not base64 encoded)
4. Include error details in responses to help with troubleshooting
5. For private repositories, ensure the token has the correct permissions

Maintain a professional tone and provide clear, actionable responses.
"""

# Define the MCP server with token in the environment
github_server = MCPServerStdio(
    "docker",
    args=[
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
    ],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PERSONAL_ACCESS_TOKEN,
    },
)

# Initialize the agent with a list to hold conversation history
conversation_history = []

github_agent = Agent(
    model,
    system_prompt=system_prompt,
    mcp_servers=[github_server],
    name="GitHub Agent"
    # message_handler=add_messages  # Use langgraph's message handler
)

# Retry mechanism for GitHub agent operations
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_with_retry(agent, input_text):
    try:
        return await agent.run(input_text)
    except Exception as e:
        if "rate limit" in str(e).lower():
            print("Rate limit hit, waiting before retry...")
        raise

# Updated main function with retry mechanism and better error handling
async def main():
    async with github_agent.run_mcp_servers():
        while True:
            try:
                user_input = input("\n[You] ")
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("Goodbye!")
                    break

                # Prepend conversation history to the input if needed:
                if conversation_history:
                    context = "\n".join(conversation_history)
                    combined_input = f"{context}\n[You]: {user_input}"
                else:
                    combined_input = user_input

                # Run the agent with retry mechanism
                try:
                    result = await run_with_retry(github_agent, combined_input)
                except Exception as e:
                    print("An error occurred while processing your request:", e)
                    continue

                # Add this round to the conversation memory
                conversation_history.append(f"[You]: {user_input}")
                conversation_history.append(f"[Assistant]: {result.data}")

                print("[Assistant]", result.data)

            except Exception as outer_e:
                print("An unexpected error occurred:", outer_e)

if __name__ == "__main__":
    asyncio.run(main())