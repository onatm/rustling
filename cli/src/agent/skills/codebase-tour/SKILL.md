---
name: codebase-tour
description: Explain how a project, or one feature of it, is put together. Use when asked what a codebase does, where something lives, or how a feature works end to end.
---

# Codebase tour

Work from the code, not from guesses. The `explore` subagent does the searching and reading, so the files it goes through stay out of your context.

1. Hand the question to `explore` with the `task` tool. It sees nothing of this conversation, so write the prompt in full:
   - the question, in the user's terms;
   - what you already know: names, files or commands the user mentioned;
   - what to report: the main parts in the order things happen, and the key places as `path:line`.

   For a whole project, ask for its shape: the manifests and README, the entry points, the main modules. For a feature, ask it to start where the user meets it (a command, an endpoint, a UI action) and follow the calls.
2. If the question has separate parts, such as two features or a client and a server, give each its own task call.
3. Read a file yourself only to check something a report left unclear.

If `task` is refused because you are a subagent yourself, do the searching directly: `glob` for the layout and manifests, `grep` for names, then read only the parts that matter, with `offset` and `limit` for long files.

Answer with:

- **What it is**: one or two sentences.
- **How it fits together**: the main parts and how data or control moves between them, in the order it happens.
- **Where to look**: the key files and functions as `path:line`, each with a few words on its role.

Say plainly what was not checked.
