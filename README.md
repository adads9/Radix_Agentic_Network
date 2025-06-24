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
  - `agents/gh_ag.py` (github_agent)
  - `agents/radix_validate.py` (radix_linter_agent)

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