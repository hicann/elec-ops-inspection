# <Project Name> Entrypoints

New Codex sessions should restore context in this order:

1. `<kb-root>/registry.yaml`
2. `<project-kb>/project-summary.md`
3. `<project-kb>/current-status.md`
4. `<project-kb>/entrypoints.md`

Load `<project-kb>/decisions.md` only when the task involves existing
decisions, path conventions, repeated judgments, or knowledge-base maintenance.

After reading these files, inspect the live repository files relevant to the
task.
