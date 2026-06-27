# Maintenance Checklist

Use this checklist before committing knowledge-base changes.

## Structure

- `registry.yaml` points to existing `kb` directories.
- Each bootstrap file listed in the registry exists.
- Project packages include `project-summary.md`, `current-status.md`,
  `entrypoints.md`, and `decisions.md`.
- Paths are relative to `.agents/knowledge` unless the file says otherwise.

## Content

- Notes are public, reviewable, and connected to repository evidence.
- Status entries are concise and dated.
- Decisions include decision, reason, and impact.
- Knowledge entries preserve durable context; task execution workflows belong in
  `.agents/skills`, not in `.agents/knowledge`.
- Claims about performance, accuracy, hardware, datasets, or production use are
  backed by source evidence or written as required evidence.

## Privacy

- No personal profiles or cross-workspace memories.
- No tokens, cookies, private keys, passwords, or account credentials.
- No private datasets, private URLs, or local generated logs.
- No copied content from a personal knowledge base except generic framework
  structure and templates.

## Validation

```bash
python .agents/knowledge/tools/validate_kb.py --root .agents/knowledge
git diff --check
```

Perform a changed-file secret scan before PR submission.

## Differentiation

- This scaffold is not another skill authoring workflow.
- It should not duplicate operator readiness, benchmark evidence, or skill
  creation instructions.
- Its PR value is durable context restoration, evidence discipline, templates,
  and offline validation.
