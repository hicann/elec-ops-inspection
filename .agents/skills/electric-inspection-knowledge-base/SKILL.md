---
name: electric-inspection-knowledge-base
description: Use when maintaining or using the repo-local Codex knowledge base for elec-ops-inspection, including .agents/knowledge registry updates, bootstrap files, public project context, durable decisions, templates, validation, and PR preparation without importing personal knowledge content.
---

# Electric Inspection Knowledge Base

Maintain and use the repo-local knowledge scaffold for
`cann/elec-ops-inspection`.

## Scope

Use this skill for:

- Restoring repository context through `.agents/knowledge/registry.yaml`.
- Adding or updating public project, domain, learning, or reference packages.
- Recording durable decisions that help future Codex sessions.
- Validating the knowledge scaffold before PR submission.

Do not use this skill to store personal memories, credentials, private datasets,
machine-local state, or unsupported claims.

Do not use this skill to design new `.agents/skills` contributions or to review
operator PR evidence. Use `electric-inspection-skill-creator` for skill
authoring and a task-specific operator review skill for PR readiness.

## Workflow

### Step 1: Load The Registry

Read `.agents/knowledge/registry.yaml` first. Choose the smallest matching entry
under `projects`, `domains`, or `learning`.

### Step 2: Read Bootstrap Files Only

Read only the bootstrap files listed for the matched entry. Load
`decisions.md` only when the task concerns existing decisions, path conventions,
repeated judgments, or knowledge-base maintenance.

### Step 3: Check Live Repository Evidence

Before recording facts, inspect the relevant live files such as:

```text
README.md
optimized_transducer/README.md
unique_v3/README.md
.agents/skills/*/SKILL.md
```

Repository files and user-provided evidence override knowledge-base notes.

### Step 4: Update The Smallest File

Use `references/maintenance-checklist.md`.

- Status or next work: update `projects/elec-ops-inspection/current-status.md`.
- Durable conventions: update `projects/elec-ops-inspection/decisions.md`.
- New repeated workflow: add or update a domain file.
- New public reference: add a short note under `references/`.
- New reusable package: start from `templates/`.

### Step 5: Validate

Run:

```bash
python .agents/knowledge/tools/validate_kb.py --root .agents/knowledge
git diff --check
```

Also scan changed files for tokens, private keys, passwords, account cookies,
and local-only paths before preparing a PR.

### Step 6: Prepare PR Text

Use `references/pr-description-template.md`. State that this PR adds a
repo-local knowledge scaffold and does not import personal knowledge content.
Also state how the knowledge scaffold differs from existing repository-local
skills: it preserves durable context and bootstrap paths, while other skills
execute task workflows.

## Error Handling

- Missing evidence: record the missing input as a requirement, not a fact.
- Private content found: remove it before validation and PR preparation.
- Duplicate package: update the existing package instead of creating another.
- Duplicate skill scope: stop and use the existing task skill instead of
  putting workflow instructions into the knowledge base.
- Validator failure: fix the registry, bootstrap path, or required project file.

## References

- [Maintenance Checklist](references/maintenance-checklist.md)
- [PR Description Template](references/pr-description-template.md)
