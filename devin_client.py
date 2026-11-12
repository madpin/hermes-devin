"""OpenAI-client-shaped facade that runs Hermes turns on Devin.

Each Hermes request is one ``devin -p`` run in a private session directory (see devin_session.py): the first
request of a conversation starts a Devin session and delivers Hermes' system prompt and tool schemas through a
``SessionStart`` hook; every later request resumes that session (``-r``) and sends only what is new (the user
message, or the tool results Hermes just produced). Resuming costs about 2 s and keeps the provider's prompt
cache warm; no process stays alive between requests.

A request whose history does not extend what the session already holds (compaction, edited history, another
system prompt, model or effort) opens a fresh session and replays the transcript.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:  # loaded as a package by Hermes, or as flat modules by tests and scripts
    from . import devin_bridge, devin_models, devin_session
except ImportError:  # pragma: no cover
    import devin_bridge
    import devin_models
    import devin_session

logger = logging.getLogger("hermes_devin")

DEFAULT_TIMEOUT = 900.0
MARKER_BASE_URL = "process://devin"


class DevinError(RuntimeError):
    """A failed Devin request. ``stale`` marks errors a fresh session may not have (the session is gone)."""

    def __init__(self, message: str, *, stale: bool = False):
        super().__init__(message)
        self.stale = stale


@dataclass
class Turn:
    text: str
    reasoning: str
    usage: dict[str, int] | None
    session_id: str | None
    metrics: tuple[int, int, int]  # cumulative (prompt, completion, cached) after this turn
    # Some models ignore "text only" and call a Hermes tool through Devin's own ``mcp_call_tool`` (Devin
    # rejects it in print mode). The intent is still clear, so it is kept: [(tool name, arguments JSON)] plus
    # the narration that came before it.
    native_calls: tuple[tuple[str, str], ...] = ()
    native_text: str = ""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _effective_timeout(timeout: Any) -> float:
    if os.environ.get("HERMES_DEVIN_TIMEOUT", "").strip():
        return _env_float("HERMES_DEVIN_TIMEOUT", DEFAULT_TIMEOUT)
    for attr in ("read", "total"):  # httpx.Timeout
        value = getattr(timeout, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    if isinstance(timeout, (int, float)) and timeout > 0:
        return float(timeout)
    return DEFAULT_TIMEOUT


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


# ================================================================================================ export
def read_export(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


_warned_tools: set[str] = set()


def _native_calls(step: dict[str, Any], tool_names: frozenset[str]) -> list[tuple[str, str]]:
    """Hermes tool calls a model made natively in one agent step (directly, or wrapped in ``mcp_call_tool``)."""
    found = []
    for call in step.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name, args = str(call.get("function_name") or ""), call.get("arguments")
        if name == "mcp_call_tool" and isinstance(args, dict):
            name, args = str(args.get("tool_name") or ""), args.get("arguments")
        if name in tool_names:
            found.append((name, args if isinstance(args, str) else json.dumps(args if args is not None else {},
                                                                              ensure_ascii=False)))
    return found


def turn_from_export(data: dict[str, Any], prev: tuple[int, int, int], stdout: str, brief_id: str | None,
                     tool_names: frozenset[str] = frozenset()) -> Turn:
    """Reply text, reasoning and this turn's token usage out of Devin's ATIF export.

    ``final_metrics`` are cumulative over the whole session, so usage is the difference from the last turn.
    With ``brief_id`` (first turn of a session) the export must show the brief arrived as system context.
    """
    steps = [s for s in data.get("steps") or [] if isinstance(s, dict)]
    if brief_id and not any(s.get("source") == "system" and brief_id in str(s.get("message") or "") for s in steps):
        raise DevinError(
            "Devin did not receive the Hermes brief (the SessionStart hook did not run or was blocked by an "
            "organisation policy). Check `devin doctor` and that HERMES_DEVIN_COMMAND points at the right CLI."
        )
    extra = {
        str((t.get("function") or {}).get("name"))
        for t in (data.get("agent") or {}).get("tool_definitions") or [] if isinstance(t, dict)
    } - devin_session.RESIDUAL_TOOLS - {"None"}
    new = extra - _warned_tools
    if new:
        _warned_tools.update(new)
        logger.warning("devin: this Devin version offers tools the plugin does not switch off: %s "
                       "(set HERMES_DEVIN_DISABLE_TOOLS to add them)", ", ".join(sorted(new)))
    last_user = max((i for i, s in enumerate(steps) if s.get("source") == "user"), default=-1)
    replies = [s for s in steps[last_user + 1:] if s.get("source") == "agent"]
    text = "\n\n".join(str(s.get("message") or "").strip() for s in replies if str(s.get("message") or "").strip())
    native: list[tuple[str, str]] = []
    narration: list[str] = []
    for s in replies:
        calls = _native_calls(s, tool_names)
        if calls:
            native += calls
            narration = [str(r.get("message") or "").strip() for r in replies[: replies.index(s) + 1]
                         if str(r.get("message") or "").strip()]
    thoughts = "\n".join(str(s.get("reasoning_content") or "").strip() for s in replies if s.get("reasoning_content"))
    fm = data.get("final_metrics") or {}
    cur = (int(fm.get("total_prompt_tokens") or 0), int(fm.get("total_completion_tokens") or 0),
           int(fm.get("total_cached_tokens") or 0))
    delta = tuple(c - p if c >= p else c for c, p in zip(cur, prev))
    usage = {"prompt": delta[0], "completion": delta[1], "cached": delta[2]} if any(cur) else None
    return Turn(text or stdout.strip(), thoughts, usage, data.get("session_id"), cur,
                tuple(native), "\n\n".join(narration))


# ============================================================================================== session
class DevinSession:
    """One Devin session bound to a (model, system prompt + tools) configuration."""

    def __init__(self, client: "DevinClient", cfg_key: str, target: devin_models.Target, brief: str,
                 brief_id: str, *, ephemeral: bool = False):
        self.client, self.cfg_key, self.target, self.brief, self.brief_id = client, cfg_key, target, brief, brief_id
        self.ephemeral = ephemeral
        self.lock = threading.Lock()
        self.canon: list[list[Any]] = []  # canonical form of every Hermes message the session already holds
        self.last_used = time.monotonic()
        self.root: Path | None = None
        self.sid: str | None = None  # Devin's session id, known after the first turn
        self.metrics: tuple[int, int, int] = (0, 0, 0)
        self.turns = 0

    @property
    def usable(self) -> bool:
        return self.sid is not None and self.root is not None and self.root.exists()

    def ask(self, text: str, timeout: float, tool_names: frozenset[str] = frozenset()) -> Turn:
        if self.root is None or not self.root.exists():
            key = uuid.uuid4().hex[:16]
            self.root = devin_session.write_session_dir(
                self.client.sessions_dir, key, self.brief, self.client.extra_disabled)
            self.sid, self.metrics, self.turns = None, (0, 0, 0), 0
        self.turns += 1
        prompt = self.root / f"turn-{self.turns}.prompt.txt"
        export = self.root / f"turn-{self.turns}.export.json"
        prompt.write_text(text, encoding="utf-8")
        argv = [self.client.command, "-p", "--prompt-file", str(prompt), "--model", self.target.uid,
                "--respect-workspace-trust", "false", "--export", str(export)]
        if self.sid:
            argv += ["-r", self.sid]
        out, err, rc = self.client.run(argv, self.root / "work", devin_session.child_env(self.root), timeout)
        data = read_export(export)
        for leftover in (prompt, export):  # transcripts: do not keep them lying around
            try:
                leftover.unlink()
            except OSError:
                pass
        if rc != 0:
            tail = (err or out).strip()[-600:]
            if "No session found" in tail:
                raise DevinError("Devin lost the session", stale=True)
            if "Not logged in" in tail or "auth login" in tail:
                raise DevinError("Devin is not logged in. Run `devin auth login`, then try again.")
            raise DevinError(f"devin exited with status {rc}" + (f": {tail}" if tail else ""))
        if data is None and not out.strip():
            raise DevinError("Devin finished without an answer" + (f": {err.strip()[-300:]}" if err.strip() else ""))
        turn = turn_from_export(data or {}, self.metrics, out, self.brief_id if self.sid is None else None, tool_names)
        if not turn.text and not turn.native_calls and not turn.usage:
            raise DevinError("Devin finished without an answer")
        self.sid = turn.session_id or self.sid
        self.metrics = turn.metrics
        self.last_used = time.monotonic()
        return turn

    def close(self) -> None:
        root, self.root = self.root, None
        if root:
            shutil.rmtree(root, ignore_errors=True)


# ================================================================================================ client
class DevinClient:
    """Drop-in for the OpenAI client Hermes builds for ``external_process`` providers."""

    HERMES_SKIP_TRANSPORT_WRAP = True  # a complete client already: never re-dispatch through a wire adapter
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 default_headers: dict[str, str] | None = None, command: str | None = None,
                 args: list[str] | None = None, **_: Any):
        self.api_key = api_key or "devin"
        self.base_url = base_url or MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self.command = (command or os.environ.get("HERMES_DEVIN_COMMAND") or os.environ.get("DEVIN_CLI_PATH")
                        or shutil.which("devin") or str(Path.home() / ".local" / "bin" / "devin"))
        self.sessions_dir = _hermes_home() / "cache" / "devin-sessions"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        self.max_sessions = int(_env_float("HERMES_DEVIN_MAX_SESSIONS", 8))
        self.idle_ttl = _env_float("HERMES_DEVIN_IDLE_SECONDS", 1800.0)
        self.extra_disabled = tuple(t for t in os.environ.get("HERMES_DEVIN_DISABLE_TOOLS", "").replace(",", " ").split() if t)
        self._sessions: "collections.OrderedDict[str, DevinSession]" = collections.OrderedDict()
        self._mutex = threading.RLock()
        self._procs: set[subprocess.Popen] = set()
        self._prune_stale_dirs()

    # ----------------------------------------------------------------------------------- housekeeping
    def _prune_stale_dirs(self) -> None:
        """Leftovers of a crashed run: session dirs untouched for a day."""
        try:
            for d in self.sessions_dir.iterdir():
                if d.is_dir() and time.time() - d.stat().st_mtime > 86400:
                    shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

    def close(self) -> None:
        self.is_closed = True
        with self._mutex:
            sessions, self._sessions = list(self._sessions.values()), collections.OrderedDict()
            procs = list(self._procs)
        for proc in procs:
            self._kill(proc)
        for sess in sessions:
            sess.close()

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                return
            try:
                proc.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                continue

    # ------------------------------------------------------------------------------------- the child
    def run(self, argv: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[str, str, int]:
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True)
        except OSError as exc:
            raise DevinError(
                f"Could not start the Devin CLI {argv[0]!r}: {exc}. Install it from https://cli.devin.ai, run "
                "`devin auth login`, or set HERMES_DEVIN_COMMAND."
            ) from exc
        with self._mutex:
            self._procs.add(proc)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            raise DevinError(f"Devin did not answer within {timeout:.0f}s") from None
        except BaseException:
            self._kill(proc)
            raise
        finally:
            with self._mutex:
                self._procs.discard(proc)
        return out, err, proc.returncode

    # --------------------------------------------------------------------------------------- sessions
    def _acquire(self, cfg_key: str, factory, canon: list[list[Any]]) -> tuple[DevinSession, int, bool]:
        """``(session, already_conveyed, ephemeral)`` with the session's lock HELD (release it when done).

        Reuses the session when the history extends what it holds; otherwise starts a new one. If the session
        is busy with another request, a private one-shot session is returned instead.
        """
        with self._mutex:
            now = time.monotonic()
            for key, old in list(self._sessions.items()):  # idle sessions go first
                if now - old.last_used > self.idle_ttl and not old.lock.locked():
                    self._sessions.pop(key, None)
                    old.close()
            sess = self._sessions.get(cfg_key)
            if sess is not None:
                n = len(sess.canon)
                if sess.lock.locked():
                    new = factory(ephemeral=True)
                    new.lock.acquire()
                    return new, 0, True
                if sess.usable and 0 < n < len(canon) and canon[:n] == sess.canon:
                    sess.lock.acquire()
                    self._sessions.move_to_end(cfg_key)
                    return sess, n, False
                self._sessions.pop(cfg_key, None)  # gone, or the history diverged: start over
                sess.close()
            while len(self._sessions) >= max(1, self.max_sessions):
                victim = next(iter(self._sessions.values()))
                if victim.lock.locked():
                    break
                self._sessions.popitem(last=False)
                victim.close()
            new = factory(ephemeral=False)
            new.lock.acquire()
            self._sessions[cfg_key] = new
            return new, 0, False

    # ---------------------------------------------------------------------------------------- request
    @staticmethod
    def _effort(kwargs: dict[str, Any]) -> str | None:
        value = kwargs.get("reasoning_effort")
        extra = kwargs.get("extra_body") or {}
        if not value and isinstance(extra, dict):
            reasoning = extra.get("reasoning") or {}
            if isinstance(reasoning, dict):
                value = reasoning.get("effort") or ("none" if reasoning.get("enabled") is False else None)
        value = value or os.environ.get("HERMES_DEVIN_EFFORT") or None
        return str(value).lower() if value and str(value).lower() in devin_models.EFFORT_ORDER else None

    def _create_chat_completion(self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
                                timeout: Any = None, tools: list[dict[str, Any]] | None = None,
                                tool_choice: Any = None, stream: bool = False, **kwargs: Any) -> Any:
        system_text, convo = devin_bridge.split_system(list(messages or []))
        if not convo:
            raise DevinError("Nothing to send: the conversation has no user message")
        effort = self._effort(kwargs)
        try:
            target = devin_models.resolve(model, effort, devin_models.catalog(self.command))
        except devin_models.UnknownModel as exc:
            raise DevinError(str(exc)) from exc
        cfg_key = hashlib.sha256(json.dumps(
            [target.uid, system_text, devin_bridge.tool_sections(tools, tool_choice)], ensure_ascii=False,
            default=str).encode()).hexdigest()
        brief_id = cfg_key[:12]
        brief = devin_bridge.build_brief(system_text, tools, tool_choice, brief_id)
        logger.debug("devin request: model=%s effort=%s tools=%d messages=%d stream=%s",
                     target.uid, effort, len(tools or []), len(messages or []), stream)
        canon = [devin_bridge.canonical(m) for m in convo]
        used_ids = devin_bridge.used_tool_ids(convo)
        timeout_s = _effective_timeout(timeout)
        tool_names = frozenset(
            str((t.get("function") or {}).get("name")) for t in tools or [] if isinstance(t, dict))

        turn: Turn | None = None
        for attempt in range(2):  # 1: the existing session; 2: a fresh one after it went away
            sess, done, ephemeral = self._acquire(
                cfg_key, lambda ephemeral: DevinSession(self, cfg_key, target, brief, brief_id, ephemeral=ephemeral),
                canon)
            try:
                text = devin_bridge.render_turn(convo[done:], fresh_history=(done == 0 and len(convo) > 1))
                turn = sess.ask(text, timeout_s, tool_names)
                break
            except DevinError as exc:
                self._drop(sess)
                if exc.stale and attempt == 0 and done:
                    logger.info("devin: session went away (%s), retrying on a fresh one", exc)
                    continue
                raise
            except BaseException:
                self._drop(sess)
                raise
            finally:
                sess.lock.release()
        assert turn is not None

        calls, cleaned = devin_bridge.extract_tool_calls(turn.text, used_ids)
        if not calls and turn.native_calls:  # see Turn.native_calls: keep the intent, drop the apology after it
            used = set(used_ids)
            for i, (name, args) in enumerate(turn.native_calls, 1):
                cid = f"devin_call_{i}"
                while cid in used:
                    cid += "_"
                used.add(cid)
                calls.append(devin_bridge.build_tool_call(cid, name, args))
            cleaned = turn.native_text
            logger.info("devin: model called %s natively; passing it on as a Hermes tool call",
                        ", ".join(n for n, _ in turn.native_calls))
        reply = {"role": "assistant", "content": cleaned, "tool_calls": calls}
        if ephemeral:
            self._drop(sess)
        else:
            sess.canon = canon + [devin_bridge.canonical(reply)]
        return self._completion(target, turn, cleaned, calls, stream)

    def _drop(self, sess: DevinSession) -> None:
        with self._mutex:
            if self._sessions.get(sess.cfg_key) is sess:
                self._sessions.pop(sess.cfg_key, None)
        sess.close()

    @staticmethod
    def _completion(target: devin_models.Target, turn: Turn, cleaned: str, calls: list[Any], stream: bool) -> Any:
        usage = None
        if turn.usage:
            prompt, out, cached = turn.usage["prompt"], turn.usage["completion"], turn.usage["cached"]
            usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out,
                                    prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
        message = SimpleNamespace(content=cleaned, tool_calls=calls, reasoning=turn.reasoning or None,
                                  reasoning_content=turn.reasoning or None, reasoning_details=None)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
                                     usage=usage, model=target.display)
        if stream:
            try:
                from agent.acp_openai_bridge import completion_to_stream_chunks

                return completion_to_stream_chunks(completion)
            except ImportError:  # pragma: no cover
                return completion
        return completion

    # ------------------------------------------------------------------------------------------ models
    def list_models(self, *, timeout_seconds: float = 30.0) -> list[str]:
        return devin_models.picker_ids(devin_models.catalog(self.command))
