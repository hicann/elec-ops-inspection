# elec-ops-inspection Knowledge Scaffold

This directory is a repo-local Codex knowledge scaffold for
`cann/elec-ops-inspection`.

It provides structure for durable, reviewable project context without importing
the contents of any personal knowledge base. Keep notes small, factual, and tied
to public repository evidence or user-provided task context.

## How To Use

1. Read `registry.yaml`.
2. Choose the matching `projects`, `domains`, or `learning` entry.
3. Read only the bootstrap files listed for that entry.
4. If a task changes project status or durable decisions, update the relevant
   project package file.

## What Belongs Here

- Repository-level context that helps future Codex sessions resume work.
- Public operator documentation reading paths.
- Decisions about how this repository organizes Codex skills and knowledge.
- Templates for adding new project, domain, or learning packages.

## What Does Not Belong Here

- Personal user profiles or cross-workspace memories.
- Private datasets, logs, access tokens, account credentials, or local secrets.
- Claimed accuracy, latency, throughput, or hardware support without evidence.
- Generated build outputs, caches, or environment-specific state.

## Validation

Run from the repository root:

```bash
python .agents/knowledge/tools/validate_kb.py --root .agents/knowledge
```
