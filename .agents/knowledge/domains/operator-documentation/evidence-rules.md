# Evidence Rules

Use these rules before recording knowledge or drafting PR text.

## Allowed Without Extra Evidence

- Public repository paths and file names.
- The existence of operators, skills, templates, or docs in this checkout.
- Task status that is limited to the current PR branch.

## Requires Source Evidence

- Accuracy, latency, throughput, memory, or speedup numbers.
- Hardware, CANN version, driver, firmware, or runtime compatibility.
- Dataset coverage, real deployment status, or production readiness.
- Claims that an operator fully matches a framework API.

## How To Record Unverified Claims

Write them as requirements or open questions, not facts. Prefer:

```text
Required evidence: benchmark logs for Ascend 910B with workload shape ...
```

Avoid:

```text
The operator is production-ready and accelerates all workloads.
```

## Privacy And Hygiene

- Never store credentials, tokens, account cookies, private URLs, or secrets.
- Do not copy private user knowledge into this repository.
- Keep local absolute paths out of committed notes unless they are only examples.
- Do not commit scratch files from `inbox/` unless they are cleaned and reviewed.
