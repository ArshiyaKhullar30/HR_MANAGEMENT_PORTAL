# ADR 002 — src-first: modules are the factory, notebooks are the lab

- **Status:** Accepted
- **Date:** 2026-08-28
- **Amends:** Step 17 of `HR_AI_Project_Build_Notes.docx`
- **Risk:** R5

## Context

The Build Notes write pipeline logic in notebook cells across Days 1–3, then
refactor it into modules on Day 4 (Step 17). By that point the notebooks and the
modules typically disagree, and "refactor" becomes "rewrite and re-verify
everything" — under time pressure, at the least forgiving moment in the build.

## Decision

`src/hrai/` is the factory. `notebooks/` is the lab.

- Every notebook **imports** from `src/hrai/`. No notebook defines pipeline logic
  inline.
- A notebook's job is narrative, charts and evidence — not implementation.
- `app/` contains the service layer and re-exports shared utilities rather than
  duplicating them, so there is exactly one implementation of anything.

## Consequences

- Step 17 becomes a checkpoint rather than a rewrite.
- Every function is unit-testable the day it is written, so Step 22 is not a
  retrofit and the 80% coverage gate is reachable.
- The API and the notebooks call the same code, so the dashboard cannot silently
  disagree with the analysis.
- Re-running any step is `make <target>` — reproducible, not a manual
  cell-by-cell ritual.
- Cost: a small amount of up-front ceremony per step before the first chart
  appears. Accepted.

Nothing about *what* is built, *when*, or *with which libraries* changes. All 29
Build Notes steps and their order stand.
