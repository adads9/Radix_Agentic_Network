from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from langgraph.graph.message import add_messages

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4-0125-preview")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")

# Prompt user for token (visible input here; in real usage consider getpass fallback if possible)
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

system_prompt = """
You are the GitHub Agent, connected to GitHub via MCP. Your responsibilities include performing code integration tasks and answering repository-related queries.
You have access to the following tools: commit, pull request, repository listing, issue listing, and code search.
If you receive instructions that you cannot perform due to access restrictions, clearly inform the user.
When handling the user’s queries, first summarize the conversation history (if any), then provide a coherent answer. If an error occurs, retry or state that an error occurred and suggest a fallback.
Always work in a professional tone and strive to never return an error code.
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
    # message_handler=add_messages  # Use langgraph's message handler
)

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

                # Run the agent with a retry mechanism
                try:
                    result = await github_agent.run(combined_input)
                except Exception as e:
                    result = await github_agent.run(f"Retry: {user_input}")
                
                # Add this round to the conversation memory
                conversation_history.append(f"[You]: {user_input}")
                conversation_history.append(f"[Assistant]: {result.data}")

                print("[Assistant]", result.data)

            except Exception as outer_e:
                print("An unexpected error occurred:", outer_e)

if __name__ == "__main__":
    asyncio.run(main())