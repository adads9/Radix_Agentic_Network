from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from langgraph.graph.message import Messages

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

llm = AzureChatOpenAI(
    model=os.getenv("GITHUB_LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("BASE_LLM_API_VERSION"),
    azure_endpoint=os.getenv("BASE_FOR_LIGHTRAG"),
)

# GitHub personal access token, found in https://github.com/settings/personal-access-tokens
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

system_prompt = """
You are the GitHub Agent, connected to GitHub via MCP. 
Your primary responsibility is to interact with GitHub through the MCP server and perform various GitHub operations accurately.

Remember that some tools might need some required info (e.g repo: Repository name). 
Please ensure that you ask the user for this information, and if they want any optional info (e.g body: Issue body content (string, optional))

IMPORTANT RULES:
1. ALWAYS verify authentication before claiming lack of access
2. For file updates, ALWAYS get current content first and decode from base64
3. When updating files, ALWAYS send plain text (not base64 encoded)
4. Include error details in responses to help with troubleshooting
5. For private repositories, ensure the token has the correct permissions

Maintain a professional tone and provide clear, actionable responses.
"""

# MCP client configuration
client = MultiServerMCPClient({
  "github": {
    "command": "docker",
    "args": ["run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"],
    "transport": "stdio",
    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PERSONAL_ACCESS_TOKEN}
  }})

async def load_mcp_tools() -> list[BaseTool]:
    # Retrieve all tools from all configured MCP servers
    return await client.get_tools()

# Initialize the agent with a list to hold conversation history
mcp_tools = asyncio.run(load_mcp_tools())

github_agent = create_react_agent(
    model=llm,
    tools=mcp_tools,
    name="github_agent",
    prompt=system_prompt,
)
