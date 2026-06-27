# AGENTS.md

## Repo-Local Knowledge Base

This repository contains a Codex-oriented knowledge scaffold under
`.agents/knowledge`.

When a task involves repository context, operator documentation, local skill
authoring, long-running work, recovering previous progress, or knowledge-base
maintenance:

1. Read `.agents/knowledge/registry.yaml` first.
2. Load only the matching bootstrap files listed in the registry.
3. Treat repository source code, README files, tests, and this `AGENTS.md` as
   stronger than notes in `.agents/knowledge`.
4. Do not bulk-read or bulk-import knowledge files.

The repository knowledge base is for public, reviewable project context only.
Do not store personal notes, private datasets, credentials, account tokens,
local machine secrets, or unverifiable performance claims in it.
