# elec-ops-inspection Entrypoints

New Codex sessions should restore context in this order:

1. `.agents/knowledge/registry.yaml`
2. `.agents/knowledge/projects/elec-ops-inspection/project-summary.md`
3. `.agents/knowledge/projects/elec-ops-inspection/current-status.md`
4. `.agents/knowledge/projects/elec-ops-inspection/entrypoints.md`

Load `decisions.md` only when the task involves existing decisions, path
conventions, repeated judgments, or knowledge-base maintenance.

After reading these files, inspect the live repository files relevant to the
task before changing source code, docs, or skills.
