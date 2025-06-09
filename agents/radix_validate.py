from typing import Any, List
import json
import subprocess
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import io

load_dotenv()

"""@tool
def validate_yaml(yaml_text: str) -> dict:
    "Run Spectral, Yamale and Radix CLI validations."
    spectral = _spectral_validate(yaml_text)

    # Yamale
    try:
        yamale.validate(yamale_schema, yamale.make_data(content=yaml_text))
        yamale_err: list[str] = []
    except yamale.YamaleError as e:
        yamale_err = [str(r) for r in e.results]

    # Radix CLI (optional – ensure `radix` binary is on PATH)
    cli_proc = subprocess.run(
        ["radix", "validate", "radix-config", "--print", "-"],
        input=yaml_text, text=True, capture_output=True
    )
    cli_err = cli_proc.stdout.strip().splitlines()

    return {"spectral": spectral, "yamale": yamale_err, "cli": cli_err}"""

# ─────────────────── 1. Azure LLM for the Linter Agent ───────────
llm = AzureChatOpenAI(
    model=os.getenv("LINTER_LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("BASE_LLM_API_VERSION"),
    azure_endpoint=os.getenv("BASE_API_BASE"),
)

# ─────────────────── 2. Spectral helper (remote JSON-Schema) ──────
SPECTRAL_IMAGE = "stoplight/spectral:latest"
SCHEMA_URL = (
    "https://raw.githubusercontent.com/equinor/radix-operator/release/json-schema/radixapplication.json"
)

def _spectral_validate(yaml_text: str) -> list[dict[str, Any]]:
    cmd = ["docker", "run", "--rm", "-i",
           "stoplight/spectral", "lint",
           "--ruleset", "file:///work/radix_ruleset.yaml",
           "--format", "json", "-"]
    proc = subprocess.run(cmd, input=yaml_text, text=True,
                          capture_output=True)
    if proc.returncode == 127:          # image missing
        raise RuntimeError("Spectral image not present")
    if proc.returncode not in (0, 2):   # any other error
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout or "[]")

    #except subprocess.SubprocessError as e:
    #    print(f"[Spectral Error] Failed to run Docker command: {e}")
    #    raise RuntimeError("Failed to run Spectral validation")

# ─────────────────── 3. Tools for the React Linter Agent ─────────
@tool
def validate_yaml(yaml_text: str) -> dict:
    """
    Validate `radixconfig.yaml` using Spectral + the JSON-schema at SCHEMA_URL only.
    Returns a dict: {"spectral": [...list of violations...]}.
    """
    return {"spectral": _spectral_validate(yaml_text)}



@tool
def apply_patch(yaml_text: str, edits: list[str]) -> str:
    """
    Apply minimal “dot-path = value” edits to fix a partially invalid file.
    Returns the patched YAML string.
    """
    from ruamel.yaml import YAML
    yaml = YAML()
    data = yaml.load(yaml_text)
    for edit in edits:
        if "=" not in edit:                      # ignore bad edit strings
            continue
        path, val = map(str.strip, edit.split("=", 1))
        keys = path.replace("[", ".").replace("]", "").split(".")
        node = data
        for k in keys[:-1]:
            k2 = int(k) if k.isdigit() else k    # list index vs key
            if isinstance(node, list):
                node = node[k2]
            else:
                node = node.setdefault(k2, {})
        leaf = int(keys[-1]) if keys[-1].isdigit() else keys[-1]
        node[leaf] = yaml.load(val)              # preserve YAML types
    out = io.StringIO()
    yaml.dump(data, out)
    return out.getvalue()

# ─────────────────── 4. ReAct Linter Agent Creation ───────────────
"""LINTER_PROMPT = (
    "You are the **Radix Linter**.\n"
    "Your task is to validate and patch a `radixconfig.yaml` file.\n"
    "You will receive a YAML file as input, and you must ensure it adheres to the Radix JSON schema.\n"
    "You will use the tools `validate_yaml` and `apply_patch` to achieve this.\n"
    
    "1. Call `validate_yaml` to get JSON errors from Spectral.\n"
    "2. If problems remain, propose the minimal dot-path patches and call `apply_patch`.\n"
    "3. Iterate at this loop until the file is valid; when clean, respond FINAL with the corrected ```yaml```.\n"
    "4. If the yaml file is invalid or causes an error when using the tools due to e.g indentation issues, you will correct this and try the validation again \n"
    "5. If you cannot fix the file, respond with a polite error message.\n"

    "If you encounter an error, please provide a clear explanation of the issue and how you attempted to resolve it.\n"
    "At the end, explain all the changes you made to the input YAML file and the reason for those changes.\n"

)"""

LINTER_PROMPT = """You are a Radix configuration validator.

TOOLS:
- validate_yaml: Check if YAML follows Radix schema
- apply_patch: Apply fixes using dot-notation paths

PROCESS:
1. Validate the input YAML
2. If issues found:
   - Analyze validation results
   - Apply necessary fixes
   - Revalidate until clean.
3. When valid, output:
   FINAL
   ```yaml
   [corrected yaml]
   ```

If validation fails or encounters errors:
- Explain the specific issues found
- Describe attempted fixes
- Suggest manual corrections needed

Focus on technical accuracy and schema compliance."""

radix_linter_agent = create_react_agent(
    model=llm,
    tools=[validate_yaml, apply_patch],
    name="radix_linter_agent",
    prompt=LINTER_PROMPT,
)

__all__ = ["radix_linter_agent"]
