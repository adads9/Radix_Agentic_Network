# Radix Agentic Network

A multi-agent system for automating Radix documentation lookup, GitHub operations, and Radix configuration validation using LangGraph, LightRAG, and Azure OpenAI.

---

## Overview

This project orchestrates three specialized agents under a supervisor using LangGraph:

- **docs_agent**: Answers all questions about Radix, generates and drafts files like `radixconfig.yaml` using up-to-date documentation.
- **github_agent**: Handles all GitHub-related actions (listing repos, querying user info, commits, PRs, etc.) via the GitHub MCP server.
- **radix_linter_agent**: Validates and patches `radixconfig.yaml` to ensure it passes Radix validation rules and schema checks.

The supervisor agent routes user requests to the correct agent(s) based on the nature of the task, ensuring seamless multi-agent collaboration.

---

## Architecture

- **Entry Point:** `agents/agent_router.py` (main supervisor and chat loop)
- **Agents:**
  - `agents/lightrag_react_agent.py` (docs_agent)
  - `agents/github_agent.py` (github_agent)
  - `agents/radix_validate.py` (radix_linter_agent)

## System Architecture

```
                           +-------------------+
                           |    Supervisor     |
                           +-------------------+
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
+-------------------+      +-------------------+      +-----------------------+
|    docs_agent     |      |   github_agent    |      |  radix_linter_agent   |
+-------------------+      +-------------------+      +-----------------------+
                                 
```

### Components & Tools

- **Supervisor (agent_router.py):**
  - Orchestrates all agent interactions and routes user requests to the appropriate agent based on the task.

- **docs_agent (lightrag_react_agent.py):**
  - Uses LightRAG for retrieval-augmented generation from Radix documentation.
  - Embedding and LLM functions powered by Azure OpenAI.
  - Tools:
    - `retrieve`: Using LightRAG For document retrieval and context-aware answering.
    - `judge_response`: Evaluate the retrivial relevancy and the groundness of the answer using Openevals LLM-As-A-Judge (Can be turned on or off through .env)

- **github_agent (github_agent.py):**
  - Connects to GitHub via the MCP server (runs in Docker).
  - Handles all GitHub-related actions (repo listing, PRs, commits, etc.).
  - Tools:
    - `load_mcp_tools`: Loads available GitHub MCP tools dynamically. Can be found on the Github MCP repo

- **radix_linter_agent (radix_validate.py):**
  - Validates and patches `radixconfig.yaml` using Radix JSON schema and style rules.
  - Tools:
    - `validate_yaml`: Check if YAML follows Radix schema
    - `apply_patch`: Apply fixes using dot-notation paths
    - `fetch_mcp`: Fetches a URL from the internet and extracts its contents as markdown (Can be turned on or off through .env)

---
---

## Features

- Natural language Q&A about Radix documentation
- Automated GitHub operations (via MCP server)
- Radix config file validation and auto-patching
- Multi-agent orchestration with clear routing policies

---

## Prerequisites

- Python 3.11 or 3.12 (recommended)
- Docker (for GitHub MCP server)
- Azure OpenAI and GitHub credentials

---

## Installation & Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Radix_Agentic_Network
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your Azure OpenAI and GitHub credentials.
   - Required variables include:
     - `API_KEY`, `SUPERVISOR_LLM_MODEL`, `BASE_LLM_API_VERSION`, `BASE_API_BASE`, `GITHUB_PERSONAL_ACCESS_TOKEN`, etc.

---

## Usage
Run the crawler file to crawl, initilize and build the knowledge base (only necassary when wanting to update the knowledge base)
```sh
python crawlers/radix_lightrag_crawl.py
```
---

---
Run the supervisor chat interface:
```sh
python agents/agent_router.py
```
Type your questions or requests. Type `exit` to quit.

---

## Notes
- For advanced configuration, see the comments in `agent_router.py` and each agent file.

---

## License
MIT License
