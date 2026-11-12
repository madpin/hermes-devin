#!/usr/bin/env python3
"""A stand-in for the `devin` CLI, just enough of it for the tests.

It keeps sessions in $DEVIN_SESSIONS_DIR, runs the SessionStart hooks of $XDG_CONFIG_HOME/devin/config.json the
way Devin does (stdin JSON in, ``additionalContext`` out), and writes an ATIF-style --export. Behaviour is chosen
by words in the prompt:
  CALL_TOOL  reply with a <tool_call> block        NATIVE  call read_file through mcp_call_tool (gets rejected)
  THINK      add reasoning                          DIE     exit 3 without answering
  HANG       never answer                           SLOW    answer after 1.5 s
  ERR        fail with a model error                NOTLOGGED  fail like a logged-out CLI
Environment: FAKE_DEVIN_LOG (JSONL of everything seen), FAKE_DEVIN_SKIP_HOOKS=1, FAKE_DEVIN_EXTRA_TOOLS=a,b.
"""
import json
import os
import subprocess
import sys
import time

LOG = os.environ.get("FAKE_DEVIN_LOG")
CATALOG = {"families": [
    {"family_label": "Claude Sonnet 5.5", "family_uid": "claude-sonnet-5-5", "slug": "claude-sonnet-5.5", "aliases": [],
     "variants": [{"model_uid": f"claude-sonnet-5-5-{e}", "label": e, "max_context_tokens": 1000000,
                   "max_output_tokens": 128000} for e in ("medium", "low", "high", "xhigh")]},
    {"family_label": "Claude Haiku 5.5", "family_uid": "claude-haiku-5-5", "slug": "claude-haiku-5.5", "aliases": [],
     "variants": [{"model_uid": f"claude-haiku-5-5-{e}", "label": e, "max_context_tokens": 1000000} for e in ("low", "medium", "high")]},
    {"family_label": "GPT-5.6 Sol", "family_uid": "gpt-5.6-sol", "slug": "gpt-5.6-sol", "aliases": [],
     "variants": [{"model_uid": f"gpt-5-6-sol-{e}", "label": e, "max_context_tokens": 1000000} for e in ("none", "low", "medium")]},
    {"family_label": "Claude Sonnet 5", "family_uid": "claude-sonnet-5", "slug": "claude-sonnet-5", "aliases": ["claude", "sonnet"],
     "variants": [{"model_uid": "claude-sonnet-5-medium", "label": "m", "max_context_tokens": 1000000}]},
    {"family_label": "SWE 1.7", "family_uid": "swe-1.7", "slug": "swe-1.7", "aliases": [],
     "variants": [{"model_uid": "swe-1-7", "label": "swe", "max_context_tokens": 200000}]},
]}


def log(**kw):
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(dict(pid=os.getpid(), **kw)) + "\n")


def arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def run_hooks(config, source, sid):
    """SessionStart hooks of the config, like Devin: stdin JSON, stdout JSON with additionalContext."""
    out = []
    if os.environ.get("FAKE_DEVIN_SKIP_HOOKS"):
        return out
    for group in (config.get("hooks") or {}).get("SessionStart") or []:
        for hook in group.get("hooks") or []:
            proc = subprocess.run(hook["command"], shell=True, capture_output=True, text=True, timeout=hook.get("timeout", 30),
                                  input=json.dumps({"hook_event_name": "SessionStart", "source": source, "session_id": sid}))
            if proc.stdout.strip():
                ctx = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
                out.append(ctx)
    return out


def main(argv):
    if argv[:2] == ["models", "list"]:
        log(event="models")
        print(json.dumps(CATALOG))
        return 0
    if "-p" not in argv:
        print("unsupported", file=sys.stderr)
        return 2
    model = arg(argv, "--model")
    known = {v["model_uid"] for f in CATALOG["families"] for v in f["variants"]}
    if model not in known:
        print(f"Error: unknown model {model!r}. Valid: {', '.join(sorted(known))}", file=sys.stderr)
        return 1
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    config = json.load(open(os.path.join(xdg, "devin", "config.json"))) if os.path.exists(os.path.join(xdg, "devin", "config.json")) else {}
    store_dir = os.environ["DEVIN_SESSIONS_DIR"]
    os.makedirs(store_dir, exist_ok=True)
    store_path = os.path.join(store_dir, "store.json")
    store = json.load(open(store_path)) if os.path.exists(store_path) else {}
    prompt = open(arg(argv, "--prompt-file")).read()
    resume = arg(argv, "-r")
    agents = os.path.join(os.getcwd(), "AGENTS.md")
    if resume and resume not in store:
        print(f"Error: No session found matching '{resume}'", file=sys.stderr)
        return 1
    sid = resume or f"fake-{len(store) + 1}-{os.getpid()}"
    sess = store.setdefault(sid, {"steps": [], "metrics": [0, 0, 0], "n": 0})
    hook_ctx = run_hooks(config, "resume" if resume else "startup", sid)
    for ctx in hook_ctx:
        sess["steps"].append({"step_id": len(sess["steps"]) + 1, "source": "system", "message": ctx})
    sys_text = "\n".join(s["message"] for s in sess["steps"] if s["source"] == "system")
    log(event="run", argv=argv, cwd=os.getcwd(), resumed=bool(resume), sid=sid, prompt=prompt, model=model,
        hook_outputs=len(hook_ctx), system_chars=len(sys_text), system=sys_text,
        env={k: os.environ.get(k) for k in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "DEVIN_SESSIONS_DIR", "DEVIN_MODEL", "NO_COLOR")},
        config=config, agents_md=open(agents).read() if os.path.exists(agents) else None)
    if "NOTLOGGED" in prompt:
        print("Not logged in. Please run `auth login` first.", file=sys.stderr)
        return 1
    if "DIE" in prompt:
        print("panic: something broke", file=sys.stderr)
        return 3
    if "HANG" in prompt:
        time.sleep(60)
    if "SLOW" in prompt:
        time.sleep(1.5)
    if "ERR" in prompt:
        print("Error: this model is disabled for your organisation.", file=sys.stderr)
        return 1
    sess["steps"].append({"step_id": len(sess["steps"]) + 1, "source": "user", "message": prompt})
    steps = []
    reasoning = "pondering the question" if "THINK" in prompt else None
    if "NATIVE" in prompt:
        steps.append({"step_id": 0, "source": "agent", "message": "Let me look.", "tool_calls": [
            {"tool_call_id": "c1", "function_name": "mcp_call_tool",
             "arguments": {"server_name": "local", "tool_name": "read_file", "arguments": {"path": "/tmp/n.txt"}}}],
            "observation": {"results": [{"source_call_id": "c1", "content": "Tool execution was rejected: User rejected this tool call"}]}})
        text = "I couldn't read it: the tool call was rejected."
    elif "CALL_TOOL" in prompt:
        text = 'I will look. <tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    else:
        text = "echo: " + prompt[-60:]
    final = {"step_id": 0, "source": "agent", "message": text, "tool_calls": []}
    if reasoning:
        final["reasoning_content"] = reasoning
    for s in steps + [final]:
        s["step_id"] = len(sess["steps"]) + 1
        sess["steps"].append(s)
    sess["n"] += 1
    sess["metrics"] = [sess["metrics"][0] + 100 + (50 if resume else 0), sess["metrics"][1] + 7, sess["metrics"][2] + (40 if resume else 0)]
    json.dump(store, open(store_path, "w"))
    tools = ["mcp_call_tool", "mcp_list_servers", "mcp_list_tools", "mcp_read_resource", "skill"]
    tools += [t for t in os.environ.get("FAKE_DEVIN_EXTRA_TOOLS", "").split(",") if t]
    export = {"schema_version": "ATIF-v1.7", "session_id": sid,
              "agent": {"name": "devin", "tool_definitions": [{"type": "function", "function": {"name": t}} for t in tools]},
              "steps": sess["steps"],
              "final_metrics": {"total_prompt_tokens": sess["metrics"][0], "total_completion_tokens": sess["metrics"][1],
                                "total_cached_tokens": sess["metrics"][2], "total_steps": len(sess["steps"])}}
    json.dump(export, open(arg(argv, "--export"), "w"))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
