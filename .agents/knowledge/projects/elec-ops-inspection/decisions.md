# elec-ops-inspection Decisions

## 2026-06-17: Keep Repo Knowledge Under .agents/knowledge

Decision: Store Codex-oriented repository knowledge under `.agents/knowledge`
instead of placing a separate personal knowledge-base layout at the repository
root.

Reason: The repository already uses `.agents/skills`, so `.agents/knowledge`
keeps Codex assets together and avoids confusing project source files with
assistant support files.

Impact: Knowledge-base paths in `registry.yaml` are relative to
`.agents/knowledge`; the repository root keeps only a lightweight `AGENTS.md`
entry point.

## 2026-06-17: Do Not Import Personal Knowledge Content

Decision: Only commit the reusable framework, templates, validator, and public
repository context.

Reason: Personal profiles, cross-workspace memories, private project notes, and
account-specific workflow details do not belong in the public repository.

Impact: Future notes must be public, reviewable, and connected to repository
evidence or explicit task context.
