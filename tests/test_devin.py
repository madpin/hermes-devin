"""Tests for the Devin provider. Run (Hermes on the path for the provider registry):

    PYTHONPATH=~/.hermes/hermes-agent uv run --no-project --python 3.14 --with openai --with pytest \
        --with ruamel.yaml python -m pytest tests -q -o addopts=

Client tests drive tests/fake_devin.py, a scripted stand-in for the CLI that runs the SessionStart hooks of the
generated config for real. Live tests (real `devin`, spends a few credits) run only with DEVIN_LIVE=1.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import devin_bridge  # noqa: E402
import devin_client  # noqa: E402
import devin_models  # noqa: E402
import devin_session  # noqa: E402
import fake_devin  # noqa: E402

FAKE = ROOT / "tests" / "fake_devin.py"


# ------------------------------------------------------------------------------------------- helpers
def user(text):
    return {"role": "user", "content": text}


def assistant(text="", calls=None):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}} for cid, name, args in calls
        ]
    return msg


def tool_result(name, text, cid="devin_call_1"):
    return {"role": "tool", "name": name, "tool_call_id": cid, "content": text}


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_DEVIN_LOG", str(log))
    for var in ("HERMES_DEVIN_EFFORT", "HERMES_DEVIN_TIMEOUT", "FAKE_DEVIN_SKIP_HOOKS", "FAKE_DEVIN_EXTRA_TOOLS",
                "XDG_DATA_HOME", "HERMES_DEVIN_DISABLE_TOOLS"):
        monkeypatch.delenv(var, raising=False)
    devin_client._warned_tools.clear()
    client = devin_client.DevinClient(command=str(FAKE))
    yield client, log
    client.close()


def runs(log: Path):
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["event"] == "run"]


def ask(client, messages, **kw):
    return client.chat.completions.create(model="claude-haiku-5-5", messages=messages, tools=kw.pop("tools", None), **kw)


# ============================================================================================= session
def test_session_dir_is_private_and_complete(tmp_path):
    root = devin_session.write_session_dir(tmp_path / "base", "k1", "BRIEF TEXT")
    mode = lambda p: oct(p.stat().st_mode & 0o777)  # noqa: E731
    assert mode(root) == "0o700" and mode(root / "xdg" / "devin") == "0o700" and mode(root / "work") == "0o700"
    for name in ("brief.txt", "hook.py", "work/AGENTS.md", "xdg/devin/config.json"):
        assert mode(root / name) == "0o600", name
    assert (root / "work" / "AGENTS.md").read_text() == devin_session.RULE
    assert (root / "brief.txt").read_text() == "BRIEF TEXT"


def test_config_switches_everything_off_and_runs_the_hook():
    cfg = devin_session.base_config("python hook.py", ("extra_tool",))
    assert set(devin_session.NATIVE_TOOLS) <= set(cfg["disabled_tools"]) and "extra_tool" in cfg["disabled_tools"]
    assert {"exec", "write", "webfetch", "run_subagent", "apply_patch", "view_image"} <= set(cfg["disabled_tools"])
    assert cfg["subagents_enabled"] is False and cfg["mcpServers"] == {} and cfg["auto_update"] is False
    imports = cfg["read_config_from"]
    assert imports["agents_standard"] is True and not any(v for k, v in imports.items() if k != "agents_standard")
    assert cfg["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "python hook.py"
    assert cfg["setup_complete"] is True


def run_hook(root, event):
    return subprocess.run([sys.executable, str(root / "hook.py")], input=json.dumps(event), capture_output=True, text=True)


def test_hook_delivers_the_brief_once_and_stays_silent_on_resume(tmp_path):
    brief = '<hermes_brief id="x">\nline "quoted" \\ back é—✓ \u2028 end\n</hermes_brief>'
    root = devin_session.write_session_dir(tmp_path, "k", brief)
    first = run_hook(root, {"hook_event_name": "SessionStart", "source": "startup", "session_id": "s1"})
    out = json.loads(first.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart" and out["additionalContext"] == brief  # exact round trip
    assert run_hook(root, {"source": "startup", "session_id": "s1"}).stdout == ""  # marker: never twice
    other = devin_session.write_session_dir(tmp_path, "k2", brief)
    assert run_hook(other, {"source": "resume", "session_id": "s1"}).stdout == ""


def test_child_env_isolates_config_but_keeps_the_login(tmp_path):
    env = devin_session.child_env(tmp_path / "root", {"HOME": "/h", "PATH": "/bin", "DEVIN_MODEL": "x",
                                                      "DEVIN_PERMISSION_MODE": "dangerous", "HTTPS_PROXY": "p"})
    assert env["HOME"] == str(tmp_path / "root" / "home") and env["XDG_CONFIG_HOME"] == str(tmp_path / "root" / "xdg")
    assert env["XDG_DATA_HOME"] == "/h/.local/share"  # where Devin keeps its login
    assert env["DEVIN_SESSIONS_DIR"] == str(tmp_path / "root" / "sessions")
    assert "DEVIN_MODEL" not in env and "DEVIN_PERMISSION_MODE" not in env and env["HTTPS_PROXY"] == "p"
    kept = devin_session.child_env(tmp_path / "root", {"HOME": "/h", "XDG_DATA_HOME": "/data"})
    assert kept["XDG_DATA_HOME"] == "/data"


# ============================================================================================= models
SAMPLE = json.dumps(fake_devin.CATALOG)


def test_parse_catalog_flattens_families():
    variants = devin_models.parse_catalog(SAMPLE)
    assert [v.uid for v in variants][:4] == ["claude-sonnet-5-5-medium", "claude-sonnet-5-5-low",
                                             "claude-sonnet-5-5-high", "claude-sonnet-5-5-xhigh"]
    v = variants[0]
    assert (v.family, v.slug, v.context, v.effort) == ("claude-sonnet-5-5", "claude-sonnet-5.5", 1_000_000, "medium")
    assert next(x for x in variants if x.uid == "swe-1-7").effort is None


@pytest.mark.parametrize("asked,effort,want", [
    ("claude-sonnet-5-5", None, "claude-sonnet-5-5-medium"),
    ("claude-sonnet-5-5", "low", "claude-sonnet-5-5-low"),
    ("claude-sonnet-5.5", "xhigh", "claude-sonnet-5-5-xhigh"),
    ("CLAUDE-SONNET-5-5", "high", "claude-sonnet-5-5-high"),
    ("claude-sonnet-5-5-low", "high", "claude-sonnet-5-5-low"),  # an exact variant already says its effort
    ("gpt-5-6-sol", "low", "gpt-5-6-sol-low"),  # family uid is gpt-5.6-sol, variants are dashed
    ("gpt-5.6-sol", None, "gpt-5-6-sol-medium"),
    ("gpt-5-6-sol", "max", "gpt-5-6-sol-medium"),  # nearest available
    ("claude-haiku-5-5", "minimal", "claude-haiku-5-5-low"),
    ("sonnet", "low", "claude-sonnet-5-medium"),  # alias of a single-variant family
    ("swe-1-7", "high", "swe-1-7"),
    ("devin", None, "claude-sonnet-5-5-medium"),  # placeholder -> default model
    ("", "low", "claude-sonnet-5-5-low"),
])
def test_resolve(asked, effort, want):
    assert devin_models.resolve(asked, effort, devin_models.parse_catalog(SAMPLE)).uid == want


def test_resolve_rejects_unknown_and_ambiguous_instead_of_guessing():
    variants = devin_models.parse_catalog(SAMPLE)
    with pytest.raises(devin_models.UnknownModel, match="Did you mean"):
        devin_models.resolve("claude-sonnet", None, variants)
    with pytest.raises(devin_models.UnknownModel, match="does not list model"):
        devin_models.resolve("gpt-9", None, variants)
    clash = variants + [devin_models.Variant("other-medium", "other", "other", ("sonnet",))]
    with pytest.raises(devin_models.UnknownModel, match="ambiguous"):
        devin_models.resolve("sonnet", None, clash)


def test_picker_and_context():
    variants = devin_models.parse_catalog(SAMPLE)
    assert devin_models.picker_ids(variants) == ["claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5-6-sol",
                                                 "claude-sonnet-5", "swe-1-7"]
    assert devin_models.context_for("swe-1-7", variants) == 200_000
    assert devin_models.context_for("nope", variants) is None


def test_catalog_probes_once_then_uses_cache(fake):
    client, log = fake
    first = devin_models.catalog(str(FAKE))
    assert "gpt-5-6-sol-low" in [v.uid for v in first]
    again = devin_models.catalog(str(FAKE))
    assert [v.uid for v in again] == [v.uid for v in first] and again[0].aliases == first[0].aliases
    assert len([r for r in log.read_text().splitlines() if '"models"' in r]) == 1


def test_catalog_falls_back_to_pinned_when_the_cli_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert devin_models.catalog("/nonexistent/devin") == list(devin_models.PINNED)
    assert devin_models.resolve("gpt-5-6-sol", "low", devin_models.PINNED).uid == "gpt-5-6-sol-low"


# ============================================================================================= bridge
def test_extract_both_shapes_and_prose_mentions():
    text = (
        'To act I emit a <tool_call> block now: '
        '<tool_call>{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{\\"x\\": 1}"}}</tool_call>'
        ' and also <tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call> done'
    )
    calls, cleaned = devin_bridge.extract_tool_calls(text)
    assert [(c.id, c.function.name) for c in calls] == [("c1", "a"), ("devin_call_2", "b")]
    assert json.loads(calls[1].function.arguments) == {"y": 2}
    assert '"name": "b"' not in cleaned and "To act I emit a" in cleaned


def test_extract_ignores_malformed_and_makes_ids_unique():
    assert devin_bridge.extract_tool_calls("<tool_call>{not json}</tool_call>")[0] == []
    calls, _ = devin_bridge.extract_tool_calls('<tool_call>{"id": "call_1", "name": "a", "arguments": {}}</tool_call>',
                                               used_ids={"call_1"})
    assert calls[0].id == "call_1_2"


def test_render_turn_shapes():
    assert devin_bridge.render_turn([user("hello")]) == "hello"
    assert devin_bridge.render_turn([tool_result("read_file", "body")]) == "Tool result (read_file):\nbody"
    hist = devin_bridge.render_turn([user("q"), assistant("a", [("c1", "t", "{}")]), user("next")], fresh_history=True)
    assert hist.startswith("Conversation so far:") and '"name": "t"' in hist and hist.endswith("continue as the Assistant.")


def test_brief_carries_prompt_tools_and_reminder():
    brief = devin_bridge.build_brief("Be kind.", TOOLS, None, "abc123")
    assert brief.startswith('<hermes_brief id="abc123">') and brief.endswith("</hermes_brief>")
    assert "Be kind." in brief and "read_file" in brief and "<tool_call>" in brief and "native" in brief
    assert "Available tools" not in devin_bridge.build_brief("Be kind.", None, None, "x")


def test_canonical_ignores_whitespace_ids_and_arg_formatting():
    ours = {"role": "assistant", "content": "I'll look.", "tool_calls": [
        devin_bridge.build_tool_call("devin_call_1", "read_file", '{"path":"/x"}')]}
    theirs = assistant("I'll   look.\n", [("other-id", "read_file", '{ "path": "/x" }')])
    assert devin_bridge.canonical(ours) == devin_bridge.canonical(theirs)
    assert devin_bridge.canonical(assistant("other")) != devin_bridge.canonical(theirs)


def test_split_system_takes_leading_system_and_developer_messages():
    text, rest = devin_bridge.split_system([{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
                                            user("hi"), {"role": "system", "content": "late"}])
    assert text == "A\n\nB" and [m["role"] for m in rest] == ["user", "system"]


# ============================================================================================= client
def test_single_turn_runs_a_bare_isolated_session(fake, monkeypatch):
    client, log = fake
    monkeypatch.setenv("DEVIN_MODEL", "should-not-leak")
    out = ask(client, [{"role": "system", "content": "You are Hermes. Secret: kiwi."}, user("hello there")], tools=TOOLS)
    msg = out.choices[0].message
    assert msg.content == "echo: hello there" and not msg.tool_calls and out.choices[0].finish_reason == "stop"
    assert out.model == "claude-haiku-5-5-medium"
    u = out.usage
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens, u.prompt_tokens_details.cached_tokens) == (100, 7, 107, 0)
    run = runs(log)[0]
    argv = run["argv"]
    assert argv[:2] == ["-p", "--prompt-file"] and argv[argv.index("--model") + 1] == "claude-haiku-5-5-medium"
    assert argv[argv.index("--respect-workspace-trust") + 1] == "false" and "--export" in argv and "-r" not in argv
    root = Path(run["cwd"]).parent
    assert Path(run["cwd"]).name == "work" and root.parent == client.sessions_dir  # private cwd, not the user's repo
    assert run["agents_md"] == devin_session.RULE and run["hook_outputs"] == 1
    assert "Secret: kiwi." in run["system"] and "read_file" in run["system"] and '<hermes_brief id="' in run["system"]
    env = run["env"]
    assert env["HOME"] == str(root / "home") and env["XDG_CONFIG_HOME"] == str(root / "xdg")
    assert env["DEVIN_SESSIONS_DIR"] == str(root / "sessions") and env["DEVIN_MODEL"] is None
    assert env["XDG_DATA_HOME"] == str(Path(os.environ["HOME"]) / ".local" / "share")
    assert "exec" in run["config"]["disabled_tools"] and run["config"]["subagents_enabled"] is False
    assert not list(root.glob("turn-*"))  # transcripts are removed after each turn


def test_second_turn_resumes_and_sends_only_the_delta(fake):
    client, log = fake
    convo = [{"role": "system", "content": "S"}, user("first question")]
    first = ask(client, convo)
    convo += [{"role": "assistant", "content": first.choices[0].message.content}, user("second question")]
    second = ask(client, convo)
    assert second.choices[0].message.content == "echo: second question"
    a, b = runs(log)
    assert not a["resumed"] and b["resumed"] and b["sid"] == a["sid"] and "-r" in b["argv"]
    assert (a["prompt"], b["prompt"]) == ("first question", "second question")
    assert b["hook_outputs"] == 0 and b["system_chars"] == a["system_chars"]  # the brief is not injected again
    u = second.usage  # usage is per turn: the difference of Devin's cumulative metrics
    assert (u.prompt_tokens, u.completion_tokens, u.prompt_tokens_details.cached_tokens) == (150, 7, 40)


def test_tool_call_roundtrip(fake):
    client, log = fake
    convo = [{"role": "system", "content": "S"}, user("please CALL_TOOL")]
    first = ask(client, convo, tools=TOOLS)
    m = first.choices[0].message
    assert first.choices[0].finish_reason == "tool_calls" and m.content == "I will look."
    assert [(c.function.name, json.loads(c.function.arguments)) for c in m.tool_calls] == [("read_file", {"path": "/tmp/x"})]
    convo += [assistant(m.content, [(m.tool_calls[0].id, "read_file", m.tool_calls[0].function.arguments)]),
              tool_result("read_file", "contents of x")]
    second = ask(client, convo, tools=TOOLS)
    assert second.choices[0].message.content.startswith("echo: ")
    assert len({r["sid"] for r in runs(log)}) == 1 and runs(log)[1]["prompt"] == "Tool result (read_file):\ncontents of x"


def test_native_call_to_a_hermes_tool_is_kept_and_the_apology_dropped(fake):
    client, log = fake
    out = ask(client, [user("please NATIVE")], tools=TOOLS)
    m = out.choices[0].message
    assert out.choices[0].finish_reason == "tool_calls" and m.content == "Let me look."
    assert [(c.function.name, json.loads(c.function.arguments)) for c in m.tool_calls] == [("read_file", {"path": "/tmp/n.txt"})]
    # no such Hermes tool: nothing to translate, the model's words come through untouched
    plain = ask(client, [user("please NATIVE, differently")], tools=None)
    assert not plain.choices[0].message.tool_calls and "rejected" in plain.choices[0].message.content


def test_diverged_history_opens_a_fresh_session_and_replays(fake):
    client, log = fake
    a = ask(client, [user("alpha")])
    ask(client, [user("alpha"), {"role": "assistant", "content": a.choices[0].message.content}, user("beta")])
    ask(client, [user("alpha EDITED"), {"role": "assistant", "content": "x"}, user("gamma")])  # compaction / edit
    rows = runs(log)
    assert rows[0]["sid"] == rows[1]["sid"] != rows[2]["sid"]
    assert rows[2]["prompt"].startswith("Conversation so far:") and "alpha EDITED" in rows[2]["prompt"]


def test_lost_session_is_replaced_transparently(fake):
    client, log = fake
    first = ask(client, [user("one")])
    root = next(iter(client._sessions.values())).root
    (root / "sessions" / "store.json").unlink()  # Devin no longer knows the session
    out = ask(client, [user("one"), {"role": "assistant", "content": first.choices[0].message.content}, user("two")])
    assert out.choices[0].message.content.endswith("continue as the Assistant.")
    rows = runs(log)
    assert rows[-1]["prompt"].startswith("Conversation so far:") and "one" in rows[-1]["prompt"] and not rows[-1]["resumed"]


def test_failures_surface_devins_own_words(fake):
    client, log = fake
    with pytest.raises(devin_client.DevinError, match="disabled for your organisation"):
        ask(client, [user("please ERR")])
    with pytest.raises(devin_client.DevinError, match=r"status 3.*panic"):
        ask(client, [user("please DIE")])
    with pytest.raises(devin_client.DevinError, match="devin auth login"):
        ask(client, [user("NOTLOGGED")])
    assert not client._sessions


def test_missing_brief_is_detected(fake, monkeypatch):
    client, log = fake
    monkeypatch.setenv("FAKE_DEVIN_SKIP_HOOKS", "1")
    with pytest.raises(devin_client.DevinError, match="did not receive the Hermes brief"):
        ask(client, [user("hello")])


def test_timeout_kills_the_child(fake):
    client, log = fake
    t0 = time.time()
    with pytest.raises(devin_client.DevinError, match="did not answer"):
        ask(client, [user("HANG")], timeout=1.5)
    assert time.time() - t0 < 15 and not client._sessions and not client._procs


def test_busy_session_gets_a_private_one_shot_session(fake):
    client, log = fake
    first = ask(client, [user("warm up")])
    convo = [user("warm up"), {"role": "assistant", "content": first.choices[0].message.content}, user("SLOW answer")]
    results = {}

    def slow():
        results["slow"] = ask(client, list(convo)).choices[0].message.content

    th = threading.Thread(target=slow)
    th.start()
    time.sleep(0.6)  # the slow request now holds the session
    results["other"] = ask(client, [user("warm up"), {"role": "assistant", "content": first.choices[0].message.content},
                                    user("a different follow-up")]).choices[0].message.content
    th.join(15)
    assert results["slow"].startswith("echo:") and results["other"].endswith("continue as the Assistant.")
    rows = runs(log)
    assert len({r["sid"] for r in rows}) == 2  # the follow-up got its own session and replayed the history
    assert rows[-1]["prompt"].startswith("Conversation so far:") or rows[-2]["prompt"].startswith("Conversation so far:")


def test_effort_selects_the_variant(fake, monkeypatch):
    client, log = fake
    ask(client, [user("a")], reasoning_effort="low")
    ask(client, [user("b")], extra_body={"reasoning": {"effort": "high"}})
    ask(client, [user("c")], extra_body={"reasoning": {"enabled": False}})
    monkeypatch.setenv("HERMES_DEVIN_EFFORT", "xhigh")
    client.chat.completions.create(model="claude-sonnet-5-5", messages=[user("d")])
    client.chat.completions.create(model="claude-sonnet-5-5-low", messages=[user("e")], reasoning_effort="high")
    assert [r["model"] for r in runs(log)] == ["claude-haiku-5-5-low", "claude-haiku-5-5-high", "claude-haiku-5-5-low",
                                               "claude-sonnet-5-5-xhigh", "claude-sonnet-5-5-low"]


def test_unknown_model_fails_loudly(fake):
    client, _ = fake
    with pytest.raises(devin_client.DevinError, match="does not list model"):
        client.chat.completions.create(model="no-such-model", messages=[user("hi")])


def test_thinking_becomes_reasoning(fake):
    client, _ = fake
    msg = ask(client, [user("THINK about it")]).choices[0].message
    assert msg.reasoning == msg.reasoning_content == "pondering the question"


def test_unexpected_tools_are_reported_once(fake, monkeypatch, caplog):
    client, _ = fake
    monkeypatch.setenv("FAKE_DEVIN_EXTRA_TOOLS", "apply_patch,teleport")
    with caplog.at_level(logging.WARNING, logger="hermes_devin"):
        ask(client, [user("one")])
        ask(client, [user("two")])
    warned = [r.getMessage() for r in caplog.records if "does not switch off" in r.getMessage()]
    assert len(warned) == 1 and "apply_patch" in warned[0] and "teleport" in warned[0]


def test_extra_tools_to_disable_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DEVIN_DISABLE_TOOLS", "teleport, other")
    client = devin_client.DevinClient(command=str(FAKE))
    try:
        assert client.extra_disabled == ("teleport", "other")
    finally:
        client.close()


def test_close_removes_session_dirs_and_stops_children(fake):
    client, log = fake
    ask(client, [user("hi")])
    root = next(iter(client._sessions.values())).root
    assert root.exists()
    client.close()
    assert not root.exists() and client.is_closed and not client._sessions


def test_stream_true_returns_chunks_with_content_and_usage(fake):
    pytest.importorskip("agent.acp_openai_bridge")
    client, _ = fake
    chunks = client.chat.completions.create(model="claude-haiku-5-5", messages=[user("hi")], stream=True)
    assert chunks[0].choices[0].delta.content == "echo: hi" and chunks[-1].usage.completion_tokens == 7


def test_bad_command_gives_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(devin_models, "catalog", lambda *a, **k: list(devin_models.PINNED))
    client = devin_client.DevinClient(command="/nonexistent/devin")
    try:
        with pytest.raises(devin_client.DevinError, match="Could not start the Devin CLI"):
            client.chat.completions.create(model="claude-haiku-5-5", messages=[user("hi")])
    finally:
        client.close()


# ========================================================================================== provider
def _load_as_package(name="devin_pkg_under_test"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_provider_registers_as_an_external_process_profile(tmp_path, monkeypatch):
    pytest.importorskip("providers")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mod = _load_as_package()
    assert mod.devin.name == "devin"
    from providers import get_provider_profile

    prof = get_provider_profile("devin")  # may be an installed copy: assert on fields, not identity
    assert prof.name == "devin" and prof.auth_type == "external_process" and prof.process_command == "devin"
    assert get_provider_profile("devin-cli") is prof
    assert prof.supports_streaming is False and prof.base_url == "process://devin"
    client = mod.devin.create_client(command=str(FAKE))  # package-relative imports worked if this builds
    assert type(client).__name__ == "DevinClient" and client.HERMES_SKIP_TRANSPORT_WRAP and client.command == str(FAKE)
    client.close()
    assert mod.devin.get_model_context_length("claude-sonnet-5-5") in (1_000_000, 200_000)
    monkeypatch.setenv("HERMES_DEVIN_CONTEXT", "123456")
    assert mod.devin.get_model_context_length("anything") == 123456
    assert "claude-sonnet-5-5" in mod.devin.fetch_models()


# ================================================================================================ live
@pytest.mark.skipif(os.environ.get("DEVIN_LIVE") != "1", reason="spends Devin credits; set DEVIN_LIVE=1")
def test_live_roundtrip_with_a_tool_and_a_resumed_second_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    client = devin_client.DevinClient()
    model = os.environ.get("DEVIN_LIVE_MODEL", "claude-haiku-5-5")
    try:
        convo = [{"role": "system", "content": "You are Hermes. The codeword is pelican-7."},
                 user("Read the file /tmp/notes.txt using the tool and reply with its contents reversed.")]
        first = client.chat.completions.create(model=model, messages=convo, tools=TOOLS, reasoning_effort="low")
        m = first.choices[0].message
        assert m.tool_calls and m.tool_calls[0].function.name == "read_file", m.content
        convo += [assistant(m.content, [(m.tool_calls[0].id, "read_file", m.tool_calls[0].function.arguments)]),
                  tool_result("read_file", "abc123")]
        second = client.chat.completions.create(model=model, messages=convo, tools=TOOLS, reasoning_effort="low")
        assert "321cba" in second.choices[0].message.content.replace(" ", "")
        assert first.usage.prompt_tokens > 0 and second.usage.prompt_tokens > 0
    finally:
        client.close()
