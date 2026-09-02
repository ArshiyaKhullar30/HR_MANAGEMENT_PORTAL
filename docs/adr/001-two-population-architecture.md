# ADR 001 — Two-population architecture; the employee datasets are not joined

- **Status:** Accepted
- **Date:** 2026-08-28
- **Supersedes:** Step 04 of `HR_AI_Project_Build_Notes.docx`
- **Findings:** F1, F8 · **Risk:** R1

## Context

The Build Notes, Step 04, specify:

> "employee_attrition joins to hr_performance_engagement on EmployeeID, one-to-one,
> because it's the same employee's performance record."

Before building to that spec we tested it, as the same document instructs:

> "Don't merge anything yet, even if two files look like they share an ID. Confirm
> the IDs actually refer to the same employees before joining — a matching column
> name isn't proof of a matching key."

## Evidence

`employee_attrition.EmployeeNumber` spans 1–2068 (1,470 employees).
`hr_performance_engagement."Employee ID"` spans 1001–4000 (3,000 employees).
They share **753** ID values. On those 753 rows:

| Attribute | Agreement | Expected if same employees |
|---|---|---|
| Gender | 48.6% | ~100% |
| Age (within 1 year) | 6.0% | ~100% |

Illustrative rows:

```
ID     attrition says                 engagement says
1001   27, Female, Lab Technician     b.1957 (66), Female, Software Engineer
1002   45, Male,   Lab Technician     b.1950 (73), Female, Software Engineer
1003   47, Female, Sales Executive    b.1973 (50), Female, Software Engineer
```

The two files are unrelated HR datasets whose ID ranges happen to overlap.

**Archive candidates (F8).** Both files in `archive/` were tested as possible
corrected replacements:

- `employee_performance_pro.csv` (500 rows, IDs 1–500) shares 377 IDs with
  `employee_attrition`; agreement is 33.4% on gender, 1.3% on exact age, 6.1% on
  department, 1.9% on job role. One shared department and one shared job role
  across the two vocabularies.
- `Employee_Performance_Dataset.csv` (5,000 rows, 6-digit IDs) has **zero** ID
  intersection with any project file.
- The two archive files do not link to each other (0 shared names).
- Neither maps to O\*NET (0/13 and 0/15 exact title matches).
- Neither contains employee-level skill data.

No substitution repairs the join; it would add a third and fourth unlinked
population.

## Decision

1. `employee_attrition` (**Population A**) and `hr_performance_engagement`
   (**Population B**) are modelled as two distinct workforce populations.
2. **No employee-level join between A and B exists anywhere in the codebase.**
   This is enforced by `scripts/check_no_cross_population_join.py`, wired into
   pre-commit and CI, and asserted by
   `tests/unit/test_no_cross_population_join.py`.
3. The two populations are bridged **only** through the role → O\*NET SOC
   crosswalk and the skill ontology built on it.
4. No file in `data/raw/` is replaced by an archive file. The brief's scope rule
   — only the datasets in `enterprise_hr_ai/data/` — stands.

## Consequences

**Positive**

- No fabricated records. Every join in the project has verified referential
  integrity (`essential_skills ⊆ occupation_data`, `software_skills ⊆
  occupation_data`).
- Population B carries its own attrition label (`EmployeeStatus =
  'Voluntarily Terminated'`, 10.8% base rate against A's 16.1%), which enables
  genuine external validation of a model trained on A — the cross-population
  transfer validation in the Copilot layer.
- The skills layer serves both populations through one ontology, so org-wide
  skill-gap coverage is 4,470 employees rather than 1,470.

**Negative**

- No single employee has both a rich attrition feature vector and an engagement
  survey score. The Employee Intelligence Table (Step 16) therefore carries a
  `population` column, and some fields are null by population.
- Attrition probability for Population B is a *transferred* prediction over the
  Common Feature Contract, not a native one. It is labelled as such in the API
  response and in the dashboard, and its measured degradation is reported.

## Alternatives rejected

- **Join anyway, on the 753 overlapping IDs.** Fabricates 753 employees and
  corrupts every downstream layer. Rejected on the evidence above.
- **Substitute an archive file.** Tested; neither is related to the project data
  (F8). Rejected.
- **Use Population B only.** Discards the clean, richly-featured, properly
  labelled dataset the attrition model needs. Rejected.
