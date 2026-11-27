---
name: codebase-tour
description: Explain how a project, or one feature of it, is put together. Use when asked what a codebase does, where something lives, or how a feature works end to end.
---

# Codebase tour

Work from the code, not from guesses. Read only what you need.

1. Get the shape first. Use `glob` for the manifests and docs at the top: `Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`, `README*`, `AGENTS.md`. Read the README and the main manifest.
2. Find the entry points: `main` functions, binaries, exported modules, routes or commands. Use `glob` for the source layout and `grep` for the names.
3. For a feature, start from where the user meets it (a command, an endpoint, a UI action) and follow the calls with `grep`, one hop at a time. Read the parts of files that matter, with `offset` and `limit` for long ones.
4. Stop when you can explain the path. Do not read every file.

Answer with:

- **What it is**: one or two sentences.
- **How it fits together**: the main parts and how data or control moves between them, in the order it happens.
- **Where to look**: the key files and functions as `path:line`, each with a few words on its role.

Say plainly what you did not check.
