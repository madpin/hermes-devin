# hermes-devin

Run [Hermes Agent](https://github.com/NousResearch/hermes-agent) on the models of your [Devin](https://devin.ai) account.

Devin's CLI is a coding agent, not a model API. This plugin drives it as a *bare model*: Hermes keeps its own agent
loop, tools, memory, skills and approvals, and Devin only supplies the model. Everything on your Devin plan is
available: Claude, GPT, Gemini, SWE and Devin's other models, with their reasoning levels.

## Requirements

- Hermes Agent 0.23.1 or newer (the version it was built and tested on), macOS or Linux
- The [Devin CLI](https://cli.devin.ai), installed and logged in (`devin auth login`)

## Install

```bash
hermes plugins install madpin/hermes-devin    # or: git clone into ~/.hermes/plugins/devin
```

## Use

```bash
hermes chat --provider devin -m claude-sonnet-5-5
```

or pick **Devin CLI (Cognition)** in `hermes model` / `/model`, or set it in `~/.hermes/config.yaml`:

```yaml
model:
  provider: devin
  default: claude-sonnet-5-5
```

Devin names every reasoning level as its own model (`claude-sonnet-5-5-low`, `-medium`, `-high`, `-xhigh`).
Give the plain name and Hermes' reasoning setting picks the level (medium by default); give an exact name and it
is used as written. Slugs (`claude-sonnet-5.5`) and Devin's aliases (`sonnet`, `opus`, `gemini`) work too.
`python devin_models.py` prints what your account offers. A name Devin does not list fails with a suggestion; the
plugin never substitutes another model.

## How it works

Devin cannot run without its own system prompt (about 17,000 characters, "You are Devin ..."), but everything else
can be switched off, so each conversation gets a private session directory and every request is one `devin -p`
run in it:

| | stock `devin -p` | what hermes-devin runs |
|---|---|---|
| system prompt | Devin's | Devin's, plus **Hermes' own as system context** |
| tools | 25 (shell, edit, web, subagents, ...) | **none usable** (20 disabled by config; the 5 `mcp_*`/`skill` tools have nothing to act on) |
| MCP servers, skills, rules, hooks | yours (about 20 s start-up with ten MCP servers) | **none** (private `HOME` and config) |
| first-turn tokens, one-word prompt | 19,832 | 8,045 |
| wall time, one-word prompt | 18 s | 2.4 s |

(Measured on the author's machine, whose Devin config has ten MCP servers, a global rules file and several skills; yours will differ.)

- **Hermes' prompt reaches the model as system-level context**, through a `SessionStart` hook that returns it as
  `additionalContext`. This is the one channel that worked: always-on rule files are cut at about 25,000
  characters (Hermes' prompt and tool list are three times that), and the same text in the first user message was
  rejected as a prompt injection by three of three models tried.
- **Tool calls** use the same text bridge as Hermes' ACP providers: Hermes' tool schemas are part of the brief, the
  model answers with `<tool_call>{...}</tool_call>` blocks, and the plugin turns them into OpenAI tool calls. Hermes
  executes them, with its usual approvals. A few models try to call a Hermes tool through Devin's `mcp_call_tool`
  instead; Devin rejects that, but the intent is read from the session export and passed on as a normal tool call.
- **Every later request resumes the session** (`-r`) and sends only what is new (the user message, or the tool
  results Hermes just produced). Through real Hermes: first call 6.0 s, next call 3.6 s, 98 % of the prompt cached.
  No process stays alive between requests.
- If the history stops extending the session (compaction, edits, a different system prompt or model), a new session
  opens and the transcript is replayed. If Devin lost the session, the request is retried on a new one.
- Token counts are real, per request (the difference of Devin's cumulative session metrics), cached tokens included.

Verified with real Hermes (file tool, random token, reversed): claude-haiku-5-5, and with `scripts/live_models.py`
(tool call, tool result, answer): claude-haiku-5-5, claude-sonnet-5-5, claude-opus-4-8, claude-sonnet-4.6, gpt-5-6-sol,
gpt-5-6-terra, gpt-5-6-luna, gpt-6-luna, gpt-5-5, gpt-5-4-mini, gemini-3-8-flash, gemini-3-5-flash, swe-2, swe-1-7 and
adaptive. Weaker models follow the text protocol less often than stronger ones; gpt-6-luna needed the
`mcp_call_tool` interception above (2 of 5 first turns produced a tool call without it; 5 of 5 full round trips
with it).

## Settings (environment)

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_DEVIN_COMMAND` / `DEVIN_CLI_PATH` | `devin` on PATH, then `~/.local/bin/devin` | Path to the Devin CLI |
| `HERMES_DEVIN_EFFORT` | unset | Reasoning effort when Hermes sends none: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `HERMES_DEVIN_DISABLE_TOOLS` | unset | Extra tool names to switch off (the plugin warns when a Devin version offers tools it does not know) |
| `HERMES_DEVIN_MAX_SESSIONS` | `8` | Sessions kept for reuse |
| `HERMES_DEVIN_IDLE_SECONDS` | `1800` | Drop a session after this long without a request |
| `HERMES_DEVIN_TIMEOUT` | Hermes' timeout, else `900` | Seconds to wait for an answer |
| `HERMES_DEVIN_CONTEXT` | from Devin's model list | Context window reported to Hermes |

## Privacy and cost

- Every message, tool result and your Hermes system prompt (memory, profile, skills) goes to Devin and the model
  provider behind it, as with any hosted model.
- Usage counts against your Devin plan like any Devin CLI run. Devin prices models per token (`devin models`);
  Devin's own prompt adds about 8,000 tokens to every request, mostly cached after the first.
- Sessions live in a private directory under `$HERMES_HOME/cache/devin-sessions/` (mode 0700, files 0600) with their
  own session store, so they never show up in your own `devin ls`; the directory is removed when the conversation
  closes. The generated brief holds Hermes' whole system prompt while the conversation is open.
- The plugin sets `HOME` and `XDG_CONFIG_HOME` for the child so your Devin config, MCP servers, skills and rules are
  not loaded. `XDG_DATA_HOME` stays real because that is where Devin keeps its own login; the plugin never reads it.
- Devin has no tools here, so a prompt injection cannot make it touch your files. What Hermes does with its own
  tools is governed by Hermes' approvals as usual. Organisation policies set in your Devin account still apply.

## Limits

- Devin's own system prompt cannot be removed. Models act as Hermes because the brief says so, and they follow it,
  but now and then a model will mention Devin.
- Text only: image parts are replaced by a note.
- Tool calling goes through text, not Devin's native tool channel.
- No token streaming: print mode delivers whole answers.
- Each request starts a process: about 2 s of overhead on top of the model, and 6 to 8 s for the first request.
- Reasoning text is only passed on when Devin exports it.
- macOS and Linux only (process groups).

## Development

```bash
PYTHONPATH=~/.hermes/hermes-agent uv run --no-project --python 3.14 --with openai --with pytest \
  --with ruamel.yaml python -m pytest tests -q -o addopts=
DEVIN_LIVE=1 ... -k live          # one real round trip (spends a few Devin credits)
python scripts/live_models.py claude-haiku-5-5 gpt-5-6-sol swe-2 --low
```

The unit tests drive `tests/fake_devin.py`, a scripted stand-in for the CLI that runs the generated
`SessionStart` hook for real.

Sibling of [hermes-amp](https://github.com/madpin/hermes-amp), which does the same for the Amp CLI.

## License

MIT
