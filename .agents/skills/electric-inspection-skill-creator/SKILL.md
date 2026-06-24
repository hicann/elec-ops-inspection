---
name: electric-inspection-skill-creator
description: Use when creating, updating, or preparing PR-ready repository-local skills for elec-ops-inspection. This skill designs differentiated .agents/skills contributions for power equipment inspection operators, Ascend/CANN operator documentation, benchmark evidence, CV inspection workflows, embodied inspection, 3D reconstruction, point-cloud preprocessing, and noisy-site voice-agent scenarios.
---

# Electric Inspection Skill Creator

Create or improve repository-local skills for `cann/elec-ops-inspection`. The goal is to turn a real power inspection operator workflow into a focused, reviewable skill contribution, not to copy a generic template.

## Principles

- Treat this repository as the source of truth. Read the root `README.md`, operator README files, and nearby code/docs before writing a skill.
- Keep each skill narrow. One skill should support one reviewer-visible workflow such as benchmark evidence, operator README readiness, 3DGS point-cloud preprocessing, or noisy-site voice-agent operator checks.
- Require differentiated value. Do not create another skill that only renames or lightly rewords an existing skill or open PR.
- Separate facts from requirements. Only claim measured accuracy, performance, hardware support, or API behavior when the repository or user-provided evidence supports it.
- Prefer progressive disclosure. Put the core workflow in `SKILL.md`; put long matrices, checklists, and templates under `references/`.

## Workflow

### Step 1: Read The Target Context

Read the smallest set of files that defines the contribution:

```text
README.md
optimized_transducer/README.md
unique_v3/README.md
<operator>/README.md or docs touched by the request
.agents/skills/ if present
```

Record the repository positioning, current operator list, inspection scenario, user workflow, and any existing skill topics. If the requested skill overlaps with an existing one, update the existing skill or choose a narrower topic.

### Step 2: Select A Skill Topic

Use `references/skill-topic-matrix.md` to choose one topic. Each topic must have:

- A specific power inspection user scenario.
- A clear input set, such as operator docs, benchmark logs, accuracy tables, code paths, or PR diff.
- A concrete output, such as a readiness report, evidence checklist, benchmark review, or PR body draft.
- A validation path that can be run locally or marked as `dry_run` with a reason.

If the topic cannot be distinguished from existing repository content, stop and report `blocked: duplicate topic`.

### Step 3: Design The Skill Structure

Default to:

```text
.agents/skills/<skill-name>/
  SKILL.md
  references/
    <topic>.md
```

Add `scripts/` only for deterministic checks that can be tested locally. Add `assets/` only for templates copied into generated outputs.

### Step 4: Write The Skill

Use `references/authoring-checklist.md` while writing. A new skill must include:

- Frontmatter with `name` and a trigger-focused `description`.
- Scope and non-scope.
- Step-by-step workflow with inputs and outputs.
- Evidence rules for scenario fit, interface docs, accuracy, performance, reproducibility, and compliance.
- Failure handling for missing logs, missing NPU environment, unsupported claims, duplicate scope, or network/token issues.
- A final output format that reviewers can inspect.

### Step 5: Validate Before PR

Run checks that fit the repository and the changed files:

```bash
git diff --check
# Run a secret scan over changed files for cloud keys, private keys,
# authorization headers, and password/token/secret assignments.
```

For documentation-only skills, also perform a manual dry-run: pick one existing operator README and confirm the skill would produce a non-empty, evidence-based report. For scripts, run a representative script test.

### Step 6: Prepare PR Text

Use `references/pr-description-template.md`. The PR description must state:

- What new skill was added.
- Which inspection workflow it supports.
- How it differs from existing repository docs or skills.
- Validation commands and dry-run status.
- Known limits and unverified claims.

## Error Handling

- Missing target evidence: write the missing input as a requirement, not as a fact.
- Duplicate skill topic: stop and propose a narrower topic or an update to the existing skill.
- No NPU or benchmark environment: use `dry_run`; do not claim runtime validation.
- Token or remote access needed: use temporary credentials only and follow GitCode PR hygiene; never store tokens in remote URLs or files.
- Scope creep into operator implementation: pause and ask whether the task is still skill authoring or has become an operator code change.

## References

- [Skill Topic Matrix](references/skill-topic-matrix.md)
- [Authoring Checklist](references/authoring-checklist.md)
- [PR Description Template](references/pr-description-template.md)
