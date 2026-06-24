# Authoring Checklist

Use this checklist before committing a new `.agents/skills` contribution.

## Frontmatter

- `name` matches the skill directory.
- `description` says what the skill does and when to use it.
- The description includes repository-relevant keywords such as `elec-ops-inspection`, `power inspection`, `Ascend`, `CANN`, `operator`, `benchmark`, `CV`, `3DGS`, `point cloud`, or `voice agent`.

## Scope

- The skill has one primary workflow.
- Non-scope is explicit when the task might drift into operator implementation, benchmark execution, or issue/PR automation.
- The skill says which target repository files to read before acting.

## Workflow

- Every step has an input, action, and output.
- Reviewer-visible output is defined, such as a report, checklist, template, or patch plan.
- The workflow has a stopping rule for missing evidence or duplicate scope.

## Evidence Rules

- Scenario claims are tied to power equipment inspection.
- Accuracy and performance claims are backed by source data or marked as required inputs.
- Hardware support is not broadened beyond available repository evidence.
- Dry-run status is explicit when no NPU, benchmark logs, or datasets are available.

## Repository Hygiene

- Keep generated skill files under `.agents/skills/<skill-name>/`.
- Add a root README entry when the skill is meant to be discoverable.
- Do not add CANNBot-specific validators or governance files unless this repository adopts them.
- Do not commit build outputs, caches, logs, tokens, private datasets, or local absolute paths.

## PR Readiness

- `git diff --check` passes.
- Secret scan over changed files has no real credential hits.
- The PR description names the skill, target workflow, validation, and known limits.
- The contribution is not a duplicate of an existing skill or open PR.
