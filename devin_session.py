"""What makes a ``devin -p`` run a bare model instead of a coding agent.

Devin cannot be started without its own system prompt (about 17k characters, "You are Devin ..."), but
everything else is configurable, and the rest is switched off here:

* ``disabled_tools`` removes 20 of its 25 tools (the remaining five, ``mcp_*`` and ``skill``, ignore it; with
  no MCP servers and no skills configured they have nothing to act on);
* a private ``HOME`` and ``XDG_CONFIG_HOME`` hide the user's MCP servers, skills, rules and hooks, and private
  ``DEVIN_SESSIONS_DIR`` keeps the sessions out of the user's own session list. ``XDG_DATA_HOME`` stays the
  real one so Devin finds its own login; this plugin never reads that file;
* subagents, config imports from other tools and auto-update are off.

Hermes' system prompt and tool schemas reach the model through a ``SessionStart`` hook that returns them as
``additionalContext``. That arrives as system-level context with no size cap, unlike rule files (cut at about 25k
characters) and unlike the first user message (three of three models called it a prompt injection). A short
always-on rule in the session directory tells the model who it is and how to call Hermes' tools as text.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

# The tools Devin 3000.12 offers (25 for Claude, plus ``apply_patch`` and ``view_image`` for OpenAI models).
# The five ``mcp_*``/``skill`` tools ignore ``disabled_tools``; the list is still passed in case a later
# version honours it for them too.
NATIVE_TOOLS = (
    "ask_user_question", "browser_preview", "close_browser_preview", "edit", "exec", "find_file_by_name",
    "get_output", "grep", "kill_shell", "notebook_edit", "notebook_read", "read", "read_subagent",
    "request_scope", "run_subagent", "todo_write", "web_search", "webfetch", "write", "write_to_process",
    "apply_patch", "view_image",
    "mcp_call_tool", "mcp_list_servers", "mcp_list_tools", "mcp_read_resource", "skill",
)
RESIDUAL_TOOLS = frozenset({"mcp_call_tool", "mcp_list_servers", "mcp_list_tools", "mcp_read_resource", "skill"})

# Variables that would change what a run does; the rest of the environment (proxies, tokens) passes through.
_STRIP_ENV = (
    "DEVIN_MODEL", "DEVIN_PERMISSION_MODE", "DEVIN_HARNESS", "DEVIN_HARNESS_LEAD_ONLY", "DEVIN_HARNESS_SIDEKICK_ONLY",
    "DEVIN_SANDBOX", "DEVIN_VERBOSITY", "DEVIN_REFUSAL_FALLBACK", "DEVIN_PREFER_EXEC_TOOL", "DEVIN_SCRIPTED_TOOLS",
)

RULE = """# Session brief (this rule replaces Devin's default role)
You are NOT acting as the Devin coding agent in this session. You are the language model behind Hermes, an agent that runs on the user's machine; Hermes owns every tool.
Hermes' binding system prompt and tool list are delivered to you at session start as hook context (a <hermes_brief> block). They are system instructions from Hermes, your operator: follow them (identity, style, tool list) as if they were part of this rule.
Any native tool you may see (such as mcp_* or skill) is disabled in this session: never call it, it only produces errors.
Hermes tools do not exist as native functions: a native function call to one of them (read_file, terminal, ...) is rejected and wastes the turn. You call them by writing text. To call one, write a block of the form <tool_call>{"id": "call_1", "type": "function", "function": {"name": "<tool>", "arguments": "<JSON string>"}}</tool_call> in your reply, then stop and wait: Hermes runs it and sends the result in the next user message, labelled "Tool result". Use only the tools listed in the brief. If no tool is needed, answer directly.
"""

# Runs at every session start or resume; speaks only on the first start. Stdlib only.
HOOK = '''import json, os, sys

event = json.load(sys.stdin)
here = os.path.dirname(os.path.abspath(__file__))
marker = os.path.join(here, "brief.sent")
if event.get("source") in ("resume", "compact") or os.path.exists(marker):
    sys.exit(0)  # already in the conversation: say nothing
with open(os.path.join(here, "brief.txt"), encoding="utf-8") as fh:
    brief = fh.read()
open(marker, "w").close()
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": brief}}))
'''


def base_config(hook_command: str, extra_disabled: tuple[str, ...] = ()) -> dict:
    return {
        "disabled_tools": list(dict.fromkeys(NATIVE_TOOLS + tuple(extra_disabled))),
        "subagents_enabled": False,
        "read_config_from": {  # only AGENTS.md, from the private working directory
            "cursor": False, "windsurf": False, "claude": False, "copilot": False, "codex": False,
            "opencode": False, "zed": False, "agents_standard": True,
        },
        "mcpServers": {},
        "auto_update": False,
        "attribution": False,
        "setup_complete": True,
        "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": hook_command, "timeout": 30}]}]},
    }


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def write_session_dir(base: Path, key: str, brief: str, extra_disabled: tuple[str, ...] = ()) -> Path:
    """Create ``base/key`` with everything one Devin session needs; returns the root.

    The brief holds Hermes' whole system prompt (memory, profile, skills), so every directory is 0700 and every
    file 0600.
    """
    base = Path(base)
    root = base / key
    dirs = [root / "home", root / "xdg" / "devin", root / "sessions", root / "work"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    for d in (base, root, root / "xdg", *dirs):
        os.chmod(d, 0o700)
    hook = root / "hook.py"
    _write_private(hook, HOOK)
    _write_private(root / "brief.txt", brief)
    _write_private(root / "work" / "AGENTS.md", RULE)
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(hook))}"
    _write_private(root / "xdg" / "devin" / "config.json", json.dumps(base_config(command, extra_disabled)))
    return root


def child_env(root: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for ``devin`` inside session ``root`` (see the module docstring)."""
    env = dict(os.environ if environ is None else environ)
    real_home = Path(env.get("HOME") or Path.home())
    data_home = env.get("XDG_DATA_HOME") or str(real_home / ".local" / "share")
    for name in _STRIP_ENV:
        env.pop(name, None)
    env.update(
        HOME=str(root / "home"), XDG_CONFIG_HOME=str(root / "xdg"), XDG_DATA_HOME=data_home,
        DEVIN_SESSIONS_DIR=str(root / "sessions"), NO_COLOR="1",
    )
    return env
