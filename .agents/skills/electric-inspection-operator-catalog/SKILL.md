---
name: electric-inspection-operator-catalog
description: 当为 elec-ops-inspection 新增、迁移、重命名或下线算子，并需要维护根 README 核心算子列表、算子目录索引、文档链接、场景分类和发布状态时使用；适用于电力巡检 CANN/Ascend/NPU 算子仓库的 catalog/index 维护，不用于 benchmark 证据、示例复现或通用 PR 预检。
---

# Electric Inspection Operator Catalog

Use this skill to keep the `elec-ops-inspection` operator catalog easy to scan after operator additions, moves, renames, or documentation-only catalog fixes.

## Core Principles

- Treat the root `README.md` operator table as a repository index, not a performance claim.
- Keep one row per published or reviewable operator directory.
- Preserve the repository's existing table shape unless the user explicitly asks for a broader README restructure.
- Link and status claims must come from files in the repository or from the user's supplied PR context.
- Do not infer hardware support, benchmark speedup, accuracy, field deployment, or business value from an operator name alone.

## Inputs

Collect these before editing:

- Root `README.md`.
- Current top-level operator directories and, when present, `operators/` subdirectories.
- Each candidate operator's `README.md`, `docs/`, `doc/`, `examples/`, and build entry files.
- The user's PR context: added, moved, renamed, deprecated, or documentation-only update.
- Existing open PR context when available, to avoid duplicating another catalog update.

If the requested change also needs benchmark evidence, example reproducibility, or broad contribution readiness, route that part to the dedicated skill instead of expanding this one.

## Workflow

1. Build an operator inventory.
   - List existing README table entries.
   - List candidate operator directories from the file tree.
   - Mark missing README entries, stale entries, duplicate names, and rows whose directory or documentation link cannot be found.
   - Output: inventory notes and the exact rows that need changes.

2. Classify catalog fields.
   - Use repository evidence to assign a concise scenario label, such as `CV视觉检测`, `具身巡检`, `语音识别`, `数学通用算子`, or `通用支撑`.
   - Write a one-sentence description focused on operator function and inspection relevance.
   - Choose a conservative status label from repository evidence: `已发布`, `开发中`, `实验性`, `待补齐`, or `已下线`.
   - Output: proposed table row values with evidence path notes.

3. Draft the README update.
   - Preserve existing column order unless the user requests a new column.
   - Keep operator names in code spans and, when the current README style allows, link names to their operator README.
   - Avoid rewording unrelated project background, maintainer lists, or license sections.
   - Output: minimal README diff or a patch-ready table snippet.

4. Check link and consistency risks.
   - Verify that each changed operator name matches a directory or documented migration target.
   - Verify that any new link resolves within the repository.
   - Check that status labels do not claim release, benchmark, or production readiness without evidence.
   - Output: pass/fail checklist and unresolved questions.

5. Prepare PR-ready notes.
   - Summarize changed catalog rows.
   - List evidence files used for each row.
   - Include validation commands and any dry-run limitations.
   - Output: PR description section for catalog/index maintenance.

## Stop Or Ask

Stop and ask before editing when:

- The user asks to move operator source directories rather than only update catalog text.
- Two directories or table rows appear to refer to the same operator with different names.
- A row would require unverified claims about performance, accuracy, hardware support, or production deployment.
- The requested table schema change would affect multiple unrelated README sections.

Mark the result as `blocked` when the repository has no discoverable operator evidence and the user cannot provide the missing context.

## Validation

Run the lightweight checks that fit the change:

```bash
git diff --check
rg -n "\]\([^)]*\)" README.md .agents/skills/electric-inspection-operator-catalog
rg -n "(A[K]IA|BEGIN [A-Z ]*PRIVATE K[E]Y|GITCODE[_]TOKEN|Authorizatio[n]:|passwor[d]\s*=|t[o]ken\s*=|s[e]cret\s*=)" README.md .agents/skills/electric-inspection-operator-catalog
```

If a markdown link checker is available in the working environment, run it against `README.md` and the changed skill files.

## Output Format

Return:

- changed catalog rows
- evidence paths used
- validation commands and results
- unresolved catalog risks
- PR description snippet

## Reference

- [Catalog maintenance checklist](references/catalog-maintenance-checklist.md)
