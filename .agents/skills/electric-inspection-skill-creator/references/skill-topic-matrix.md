# Skill Topic Matrix

Read this file when choosing what repository-local skill to create for `elec-ops-inspection`.

## Topic Selection Table

| Topic | Use When | Required Inputs | Expected Output | Avoid |
|---|---|---|---|---|
| Operator README evidence check | A PR adds or rewrites an operator README | Operator README, root README, accuracy/performance tables | Readiness report covering scenario, interface, accuracy, performance, reproducibility | Rewriting the operator README without evidence |
| Inspection benchmark evidence | A PR claims latency, throughput, memory, or speedup | Hardware model, CANN version, workload shape, baseline, raw timing | Benchmark evidence checklist and PR text | Claiming unmeasured speedup |
| CV defect operator proposal | A contributor proposes an operator for visual defect detection | Defect type, model path, input/output tensors, evaluation metric | Skill workflow for checking whether the proposal is reviewable | Generic CV wording without power inspection mapping |
| Embodied inspection 3DGS data path | Work concerns drones, robot dogs, point clouds, 3DGS, or sparse graph preprocessing | Data representation, dedup/compression target, latency constraint, operator docs | Evidence checklist for spatial reconstruction or navigation workflows | Treating 3DGS as a slogan without data constraints |
| Noisy-site voice-agent operator | Work concerns ASR, RNN-T, speech instructions, or offline inspection agents | Noise scenario, sequence lengths, dtype, reference implementation, accuracy metric | Validation workflow for speech-related operator docs | Claiming robustness without dataset or metric |
| Skill contribution authoring | A contributor wants to add another `.agents/skills` contribution | Target skill topic, existing skills, docs entry, validation plan | Draft skill structure and PR-ready checklist | Duplicating an existing skill with a new name |

## Selection Rules

1. Pick one main topic per PR.
2. Prefer a topic tied to an existing operator or a clearly stated future operator workflow.
3. If a topic needs unavailable measurements, keep it as a checklist or input requirement.
4. If two topics are both important, split them into two PRs with different files and reviewer value.
5. If the only difference is wording or account ownership, do not create a new PR.

## Minimum Evidence By Topic

| Evidence Type | Minimum Requirement |
|---|---|
| Scenario | A concrete power inspection workflow, not only generic AI acceleration |
| Interface | Inputs, outputs, attrs, dtype, shape, layout, and constraints |
| Accuracy | Baseline implementation, tolerance, dtype, shape range, pass/fail status |
| Performance | Hardware, CANN version, workload, baseline, unit, repeated timing method |
| Reproducibility | Commands, examples, or clearly marked `dry_run` |
| Compliance | No secrets, generated binaries, private datasets, or unverifiable claims |
