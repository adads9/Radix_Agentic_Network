from typing import Any, List, Dict, Tuple
import sys, json, yaml, subprocess, tempfile, requests, jsonschema
import subprocess
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import yaml, json, jsonpatch, copy
from typing import List, Union, Dict
from io import StringIO
import asyncio

load_dotenv()

# ─────────────────── 1. Azure LLM + ENV variables for the Linter Agent  ───────────
llm = AzureChatOpenAI(
    model=os.getenv("LINTER_LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("BASE_LLM_API_VERSION"),
    azure_endpoint=os.getenv("BASE_API_BASE"),
)

fetch = os.getenv("FETCH_MCP")

try:
    from ruamel.yaml import YAML  # type: ignore

    _ruamel = True
    _yaml_rt = YAML()
    _yaml_rt.preserve_quotes = True
except ImportError:  # pragma: no cover – ruamel.yaml not installed
    _ruamel = False


# URL for valid JSON-Schema. Dynamically updates with new updates.
SCHEMA_URL = ("https://raw.githubusercontent.com/equinor/radix-operator/master/json-schema/radixapplication.json")
_schema_cache = None

# ─────────────────── 2. Functions for tools ─────────── 

def _schema():
    """
    Gets the schema as a json file
    """
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = requests.get(SCHEMA_URL, timeout=10).json()
    return _schema_cache


def _collect_schema_errors(yaml_text: str) -> Tuple[bool, List[str]]:
    """Validate YAML against Radix JSON‑Schema and collect **all** errors with field paths."""
    schema = _schema()
    data = yaml.safe_load(yaml_text)
    validator = jsonschema.Draft202012Validator(schema)

    def _fmt(err: jsonschema.ValidationError) -> str:
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        # remove leading slash for cleaner output
        return f"{path}: {err.message}"

    errors = sorted((_fmt(e) for e in validator.iter_errors(data)), key=str)
    return (True, []) if not errors else (False, errors)


def _rt_load(yaml_text: str) -> Any:
    if _ruamel:
        return _yaml_rt.load(yaml_text)
    return yaml.safe_load(yaml_text)


def _rt_dump(data: Any) -> str:
    if _ruamel:
        buf = StringIO()
        _yaml_rt.dump(data, buf)
        return buf.getvalue()
    return yaml.safe_dump(data, sort_keys=False)


# ─────────────────── 3. Tools for the Linter Agent  ───────────
@tool
def validate_yaml(yaml_text: str) -> str:
    """Return `{passed: bool, errors: [str]}` with full JSON‑schema error list."""
    passed, errors = _collect_schema_errors(yaml_text)
    return json.dumps({"passed": passed, "errors": errors}, indent=2)



@tool
def apply_patch(
    yaml_text: str,
    patches: Union[str, List[Union[str, Dict[str, Any], List[Dict[str, Any]]]]],
) -> str:
    """Apply RFC‑6902 JSON patches.

    The calling agent is responsible for re‑invoking `validate_yaml`.
    """

    # ---- 1. normalise patches ----
    def _json_to_ops(js: str) -> List[Dict[str, Any]]:
        obj = json.loads(js)
        if not isinstance(obj, list):
            raise ValueError("Patch JSON must be an array of operations")
        return obj

    patch_docs: List[List[Dict[str, Any]]] = []
    if isinstance(patches, str):
        patch_docs.append(_json_to_ops(patches))
    elif isinstance(patches, list):
        for item in patches:
            if isinstance(item, str):
                patch_docs.append(_json_to_ops(item))
            elif isinstance(item, dict):
                patch_docs.append([item])
            elif isinstance(item, list) and all(isinstance(op, dict) for op in item):
                patch_docs.append(item)
            else:
                raise ValueError(f"Unsupported patch item type: {type(item)}")
    else:
        raise ValueError(f"Unsupported patches type: {type(patches)}")

    # ---- 2. apply sequentially ----
    data = _rt_load(yaml_text)
    applied = 0
    for doc in patch_docs:
        data = jsonpatch.apply_patch(data, doc, in_place=False)
        applied += 1

    final_yaml = _rt_dump(data)

    return json.dumps(
        {
            "yaml": final_yaml,
            "patches_applied": applied,
        },
        indent=2,
    )


# ---------- MCP Client for fetch (optional) ----------
client = MultiServerMCPClient({
  "fetch": {
    "command": "python",
    "args": ["-m", "mcp_server_fetch"],
    "transport": "stdio",
  }})

async def load_mcp_tools() -> list[BaseTool]:
    # Retrieve all tools from all configured MCP servers
    return await client.get_tools()


# ─────────────────── 4. Experimental ─────────────────────────
tools=[validate_yaml, apply_patch]

linter_prompt_no_fetch= """You are a Radix configuration validator.


Your task is to validate and patch a `radixconfig.yaml` file according to the Radix JSON schema. 
This includes checking for schema compliance, style violations, and applying minimal patches to ensure the file passes validation. 

TOOLS:
- validate_yaml: Check if YAML follows Radix schema
- apply_patch: Apply fixes using dot-notation paths

PROCESS:
When the user supplies YAML, "Before passing the YAML file to the tools, make sure that the YAML syntax is correct and can be parsed by the tools, and get rid of any indentation errors.

Then, first call validate_yaml.
- If validation fails, think step‑by‑step to craft RFC‑6902 JSON patches that fix the errors, then call apply_patch.
- Repeat until the file validates or no further patches can be found. 
- When it passes, return the final YAML to the user.
When valid, output:
   FINAL
   ```yaml
   [corrected yaml]
   ```

If validation fails or encounters errors:
- Explain the specific issues found
- Describe attempted fixes
- Suggest manual corrections needed

If the radixconfig includes invalid fields or missing necassary fields, make sure to explain this in the output aswell.

At the end, provide a summary of the problems that were found, how they were fixed and a list of things added/removed.

Focus on technical accuracy and schema compliance."""

linter_prompt_fetch= """You are a Radix configuration validator.


Your task is to validate and patch a `radixconfig.yaml` file according to the Radix JSON schema. 
This includes checking for schema compliance, style violations, and applying minimal patches to ensure the file passes validation. 

TOOLS:
- validate_yaml: Check if YAML follows Radix schema
- apply_patch: Apply fixes using dot-notation paths
- fetch_mcp: Fetches a URL from the internet and extracts its contents as markdown


PROCESS:
When the user supplies YAML, "Before passing the YAML file to the tools, make sure that the YAML syntax is correct and can be parsed by the tools, and get rid of any indentation errors.

Then, first call validate_yaml.
- If validation fails, think step‑by‑step to craft RFC‑6902 JSON patches that fix the errors, then call apply_patch.
- Repeat until the file validates or no further patches can be found. 
- When it passes, return the YAML.
- As a final check, fetch the radix config URL: (https://radix.equinor.com/radix-config) and make sure that the returned YAML adheres to all the radix config information, 
  without any indentation errors or incorrect/invalid fields, before returning the final YAML to the user.
- If the YAML file is incorrect, make the changes necassary before passing the final YAML file.
When valid, output:
   FINAL
   ```yaml
   [corrected yaml]
   ```

If validation fails or encounters errors:
- Explain the specific issues found
- Describe attempted fixes
- Suggest manual corrections needed

If the radixconfig includes invalid fields or missing necassary fields, make sure to explain this in the output aswell.

At the end, provide a summary of the problems that were found, how they were fixed and a list of things added/removed.

Focus on technical accuracy and schema compliance."""

if fetch=='on':
    fetch_mcp = asyncio.run(load_mcp_tools())
    tools.extend(fetch_mcp)
    linter_prompt = linter_prompt_fetch
else:
    linter_prompt=linter_prompt_no_fetch

# ─────────────────── 5. ReAct Linter Agent Creation ───────────────

radix_linter_agent = create_react_agent(
    model=llm,
    tools=tools,
    name="radix_linter_agent",
    prompt=linter_prompt,
)

__all__ = ["radix_linter_agent"]
