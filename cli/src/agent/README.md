# Rustling chat

`rustling chat` runs a coding assistant for the current directory, with the model in the same process.
Its built-in tools read files (`read`), search their contents (`grep`) and find paths (`glob`).
Each asks for approval by default. For questions that take many searches and reads, it can hand the work to an
[`explore` subagent](#subagents).

To install the CLI and download the model, follow the [project quick start](../../../README.md#quick-start).

## Start or resume a session

```sh
rustling chat                                   # a new session
rustling chat --continue                        # the latest session in this directory
rustling chat --resume                          # list this directory's sessions
rustling chat --resume 019aa1b2c3d4-5e6f7a8b    # continue one of them
rustling chat --yes                             # read and search without asking
rustling chat --effort high                     # think longer before answering
rustling chat --system "Answer in French."      # add to the agent's instructions
rustling chat --logs                            # keep the log, behind a `logs` button
rustling chat --context 32768 --max-tokens 8192 # a larger window and longer replies
```

Rustling builds the agent with manifatura. The system prompt combines its one-line instructions, the tools'
guidelines, the applicable project instructions, the working directory and the date. [Base instructions](system.md).

Its instructions follow the `AGENTS.md` convention: `~/.config/rustling/AGENTS.md`, then `AGENTS.md` (or
`CLAUDE.md`) in each directory from the repository root down to the current one. Sessions are kept per directory
under `~/.local/share/rustling/sessions`. Processed prompts are kept as `serve` keeps them (`--cache-gb`,
`--cache-dir`, `--no-cache-dir`), so a resumed session, even after a restart, starts in seconds.

## Composer and keyboard

Thinking and replies appear directly below the preceding message as they stream. The conversation stays in
the terminal's scrollback for scrolling, selecting and copying. A shaded prompt with a `›` and a blank row
above and below the text marks both your draft and your sent messages. The composer stays at the bottom as
multiline drafts grow upward. Command suggestions and logs open above the input.

Type `@` followed by part of a filename or path to search the working directory. Use ↑ or ↓ to choose a
match, then Tab or Enter to insert its relative path into your draft. Esc closes the picker. File search
respects project ignore rules. Rustling uses its read tool when it needs the contents.

Type while the model loads and press Enter when it is ready. During a turn you can send a follow-up or press
Esc to interrupt. When a tool needs approval, the composer shows the action, its target and three choices.
Use ↑ or ↓ and Enter, or press `y` to allow once, `a` to allow that tool for the session, or Esc to deny.
Your draft returns when you answer.

Successful tools occupy one line. Errors include their cause, and `/tools` shows the output from the current
or most recent turn. The footer shows reasoning effort and context usage. Use `/status` for the full session
ID, prompt reuse, generation speed and loaded instruction files.

There is no log unless you ask for one with `--logs` (at `--log-level`). Then a `logs` button at the end of the
footer opens it; click it, or press Ctrl-L. The button needs the mouse, so with `--logs` the scroll wheel goes
to rustling instead of scrolling the terminal, and selecting text needs the modifier your terminal uses for that
(Option in iTerm2 and Terminal.app, Shift in most others).

| Key | What it does |
|---|---|
| Enter, Shift+Enter | Send a message or insert a new line, respectively. Sent while the agent works, a message reaches it before its next step. |
| Esc, Ctrl-C | Stop the agent at once. Ctrl-C twice, or Ctrl-D, quits. |
| `y`, `a`, `n` | Allow once, allow that tool for the session, or deny the request. |
| Tab | Insert a selected file path or complete a command or session ID. Use Up and Down to select a suggestion and Esc to dismiss the idle menu. |
| Ctrl-L | Open or close the log, with `--logs`. |
| Up, Down | Browse sent messages. Moving past the newest message restores your draft. |

## Chat commands

Typing `/` lists the commands: `/mcp` lists the MCP servers, `/skills` lists the skills, `/explore <question>` hands a question to the [`explore` subagent](#subagents), `/new` starts a new session, `/resume` opens a picker ordered by last activity (↑/↓ to select, Enter to resume, Esc to cancel),
`/resume ID` switches directly, `/sessions` lists them, `/compact [focus]`
summarises older messages to make room, `/effort` shows or sets how much the model reasons, and `/exit` quits.
`/effort off` stops the reasoning altogether, as `--no-reasoning` does, and `/effort high` (or any level) brings
it back. The effort is part of the prompt, so the reply after a change processes the conversation again.

## Skills

A skill is a set of instructions for one kind of task, in the [Agent Skills](https://agentskills.io) format: a
directory with a `SKILL.md`. The prompt lists each skill's name and description. When a task matches one, the
model loads its instructions with the `skill` tool, which never asks for approval.

Rustling has one skill built in, `codebase-tour` ([its instructions](skills/codebase-tour/SKILL.md)), for questions
about how a project fits together. It hands the searching and reading to the [`explore` subagent](#subagents). A project adds its own under `.rustling/skills/<name>/SKILL.md`, in the
repository root or any directory down to the current one. A project skill with the same name as a built-in one
replaces it. Rustling reads only these two sources: not `.agents/skills`, and no skills from your home directory.

```markdown
---
name: release
description: Prepare a release. Use when asked to cut, tag or announce a release.
---

1. Read CHANGELOG.md and list what changed since the last version.
2. …
```

Start a skill yourself with `/name`, and add a request after it: `/codebase-tour how do sessions resume?`. Its
instructions go to the model with your message. `/skills` lists the skills and where each comes from, and
`/status` names them. Skills are read when the chat starts, so a new or changed skill applies from the next one.

## Subagents

A subagent works on one task in a session of its own and reports back. Only its report enters the chat's context,
which keeps a small window free for the conversation. The model starts one with the `task` tool.

Rustling has one built in, `explore` ([its brief](agents/explore.md)), which answers a question about the code
with what it found as `path:line`. The model may start it itself; to be sure it runs, start it yourself:

```text
/explore how does rustling decide which skills to load, and which one wins when two share a name?
```

Its report then goes to the chat with your question, and the chat answers from it. Until it reports, Esc stops it
and messages wait. It is the chat's own agent with the same tools and instructions, so the engine
reuses the prompt it has already processed, and the brief goes before the task. It may call only `read`, `grep` and
`glob` (`tools` in [its definition](agents/explore.md)): it is offered the chat's other tools, so its prompt
matches, but calls to them are refused. It cannot start subagents of its own. It has 20 model calls
(`max_steps`): from halfway on it is told which one it is on, and the last is for its report, with tool calls
off, so a search that runs long still ends with what it found. While it works, the transcript shows its latest tool call
(`Exploring where sessions resume · Search fn resume`), and the finished line counts its calls.

Its tools ask for approval as the chat's do, and the question names it (`explore · Read this file?`). Allowing a
tool for the session holds for both. Esc stops the subagent along with the turn. Its sessions are kept with the
chat's, but `--resume` and `/resume` leave them out.

## Reasoning

`--effort` is how much the model reasons before it replies: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`
or `max`, or a number from 0 to 0.99.

Even `none` leaves a little reasoning: the model was built that way, and its makers say so. To have none at all:

```sh
rustling chat --no-reasoning                    # answers start at once
```

This keeps the model from ever opening a reasoning block. Answers come sooner and are less considered, and a
model that cannot reason is less inclined to call tools: asked for the weather with a weather tool on hand, it
offered to look instead of looking. Leave it off for coding agents.

## MCP servers

The chat can use the tools of MCP servers, reached at a URL (streamable HTTP) or run on this machine (stdio), set
up in `~/.config/rustling/mcp.toml`. For [Exa](https://exa.ai)'s web search, through its hosted server:

```toml
[servers.exa]
url = "https://mcp.exa.ai/mcp"
env_headers = { x-api-key = "EXA_API_KEY" }   # optional: without a key, Exa's free rate limit applies
eager = true                                  # offer its tools directly
```

A local server is a `command`, with `args`, `env` and `cwd`; an HTTP one takes `headers` and `env_headers`.
Servers start while the model loads; one that does not start is reported and left out. A server's tools are
named `<server>__<tool>`. Without `eager` they are kept out of the prompt, which matters with a small context
window, and the model finds them with `find_tools` and runs them with `run_tool`. `approve = true` asks before
each call. `/mcp` lists the servers and their tools.

## Other coding agents

Start the local server before launching your client.

```sh
rustling serve --context 32768
```

The server speaks `POST /v1/chat/completions` and `POST /v1/responses`, with streaming and tool calls. An agent's
opening prompt is long (10,000 tokens and more) and takes a minute or two the first time. The server keeps what
it has processed, in memory and under `~/.cache/rustling`, so after that a new session starts in a few seconds,
also after a restart.

Give the agent a long timeout: nothing is sent while a long prompt is being processed.

**Codex**

```toml
# ~/.codex/config.toml
[model_providers.rustling]
name = "Rustling"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1800000
stream_max_retries = 0
request_max_retries = 0

[profiles.rustling]
model_provider = "rustling"
model = "inkling-2-mini"
```

```sh
codex exec --profile rustling "Read the README and summarise this project."
```

**aider**

```sh
aider --model openai/inkling-2-mini --openai-api-base http://127.0.0.1:8080/v1 --openai-api-key rustling --timeout 1800
```

**opencode** and anything else that takes an OpenAI-compatible endpoint: point it at `http://127.0.0.1:8080/v1`
with the model name `inkling-2-mini`.
