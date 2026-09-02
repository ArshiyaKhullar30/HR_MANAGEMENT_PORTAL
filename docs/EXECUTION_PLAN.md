# Enterprise HR AI — Execution Plan
### Workforce Intelligence & Upskilling Platform

**Status:** Plan v1.0 — awaiting approval before build
**Author:** Engineering lead
**Source of truth:** `archive/Project_Architecture_DOCS/HR_AI_Project_Build_Notes.docx`
**Scope rule:** only the five datasets in `enterprise_hr_ai/data/raw/` may be used.
**Archive audit:** both `archive/*.csv` files tested as replacement candidates and rejected — see F8.

---

## 0. Executive summary

The Build Notes define a sound project: four working days (Data → ML → Workforce Intelligence → Application) plus an enterprise-hardening phase, across 29 numbered steps. This plan keeps that architecture, that step order, and that toolchain **exactly as written**.

It changes two things, both for stated reasons:

1. **Eight findings from a full audit of the five raw datasets (plus the two archive candidates)** are folded into the plan. One of them invalidates a join the Build Notes assume, and would have silently corrupted every downstream number if we had built to spec. The Build Notes' own note-to-self anticipated exactly this risk.
2. **A "src-first" working rule** replaces the notebooks-then-refactor sequence, so that Step 17 ("Refactor Notebook Code Into Modules") becomes a formality rather than a rewrite. Detail in §3.

It adds one headline capability — the **Retention ROI Copilot** (§7) — that turns the platform from predictive ("who will leave") to prescriptive ("what should we do about it, and is it worth the money").

---

## 1. What the audit found

Every claim below was verified directly against the raw files before this plan was written.

### F1 — CRITICAL: the two employee datasets are different companies. They must not be joined.

Build Notes Step 04 states: *"employee_attrition joins to hr_performance_engagement on EmployeeID, one-to-one, because it's the same employee's performance record."*

That is not true of this data.

| Check | Result | Reading |
|---|---|---|
| `employee_attrition.EmployeeNumber` | 1,470 IDs, range 1–2068 | — |
| `hr_performance_engagement.Employee ID` | 3,000 IDs, range 1001–4000 | — |
| Numeric ID intersection | 753 | Looks joinable |
| **Gender agrees on those 753** | **48.6%** | **A coin flip** |
| **Age agrees within 1 year** | **6.0%** | **Random** |

Sample of the "matched" rows:

```
ID    | attrition says              | engagement says
1001  | 27, Female, Lab Technician  | b.1957 (66), Female, Software Engineer
1002  | 45, Male,   Lab Technician  | b.1950 (73), Female, Software Engineer
1003  | 47, Female, Sales Executive | b.1973 (50), Female, Software Engineer
```

These are two unrelated HR datasets whose ID ranges happen to overlap. Joining them would fabricate 753 employees who do not exist and would poison the attrition risk, engagement, skill-gap, and recommendation layers simultaneously.

> The Build Notes already called this: *"Don't merge anything yet, even if two files look like they share an ID. Confirm the IDs actually refer to the same employees before joining — a matching column name isn't proof of a matching key."* We confirmed. They don't. **This plan follows that rule rather than the later assumption.**

**Resolution:** a two-population architecture, bridged by the role/skill ontology instead of by employee ID. See §2.

### F2 — `essential_skills.csv` is not technical skills

It is the O\*NET **Basic Skills** file: **10** cognitive skills only — Reading Comprehension, Critical Thinking, Active Listening, Mathematics, Science, Writing, Speaking, Monitoring, Active Learning, Learning Strategies. Each is scored per occupation on two scales: `IM` (Importance, 1–5) and `LV` (Level, 1–7).

The Build Notes' worked example — `required = {'Python','SQL','MLOps','Docker','AWS'}` — cannot come from this file. It comes from `software_skills.csv`: **8,753 distinct tools** across **134 categories**, each flagged `Hot Technology` (11,571 rows) and `In Demand` (2,425 rows).

**Resolution:** a two-tier skill taxonomy — **Foundational** (10 graded competencies) and **Technical** (8,753 tools). This is richer than the original single-tier design, not poorer.

### F3 — No employee-skill data exists in any of the five files

The Build Notes anticipated this and pre-authorised the fallback: *"If it doesn't → build a controlled table for the MVP."* Confirmed: none of the five files record what skills an individual holds.

**Resolution:** a **deterministic derivation**, not random generation — driven only by real in-data signals (`Title`, `Training Program Name`, `Training Outcome`, `Current Employee Rating`, tenure, `Education`, `EducationField`, `JobLevel`, `TrainingTimesLastYear`), seeded, versioned, and carried through the API and dashboard with an explicit `is_derived = true` flag. See §6 Step 12.

### F4 — Job roles do not match O\*NET occupations at all

- `employee_attrition.JobRole`: **0 of 9** exact matches into O\*NET titles
- `hr_performance_engagement.Title`: **0 of 32** exact matches
- `occupation_data.csv`: 1,016 O\*NET occupations

Without a crosswalk, the three O\*NET files are unusable and the entire skills half of the project has no input.

**Resolution:** a semantic Role→SOC crosswalk using sentence-transformers — the same library the Build Notes already sanction for the recommendation engine. 41 distinct roles is small enough to human-review every mapping. See §6 Step 11. This is also WOW component A.

### F5 — The engagement file is not one row per employee

3,150 rows / 3,000 distinct IDs. 150 employees appear twice (survey events). The grain is *employee × survey/training event*, not *employee*.

**Resolution:** grain is resolved explicitly and documented before any aggregation. Silent `groupby` on the wrong grain is how engagement averages quietly go wrong.

### F6 — Real data-quality defects to encode as validation rules

| Defect | Count | Where |
|---|---|---|
| `ExitDate` populated but status is Active / Future Start / Leave of Absence | **1,198 rows** | engagement |
| Trailing-whitespace category (`'Production       '`) | 2,115 rows | engagement |
| Mixed date formats (`20-Sep-19` vs `07-10-1969`) | all date cols | engagement |
| Constant columns (`EmployeeCount`, `Over18`, `StandardHours`) | 3 cols | attrition |
| Direct PII (`FirstName`, `LastName`, `ADEmail`, `Supervisor`) | 4 cols | engagement |
| Duplicate ID rows | 150 | engagement |

Also: **`Engagement Score` is on a 1–5 scale, not 0–100.** Build Notes Step 02 specifies a 0–100 range check; applied literally it would pass every row and catch nothing.

Attrition file is clean: 0 missing cells, 0 duplicate rows, target 237 Yes / 1,233 No (16.1%).

### F7 — Leakage register

In the engagement data, `ExitDate`, `TerminationType`, `TerminationDescription`, and `EmployeeStatus` **are** the label. Any of them entering a feature matrix produces a model with perfect scores and zero value. Registered and blocked at the schema level, not by memory.

### F8 — The `archive/` datasets are not corrected versions of anything. Do not substitute them.

Both archive CSVs were put through the same evidence test as F1, to check whether either is the "real" file that would make the employee join valid.

**`employee_performance_pro.csv`** (500 rows, IDs 1–500) — 377 IDs overlap with `employee_attrition`:

| Field | Agreement on the 377 shared IDs |
|---|---|
| Gender | **33.4%** (chance, for a 3-value field) |
| Age (exact) | **1.3%** |
| Department | **6.1%** |
| JobRole | **1.9%** |
| YearsAtCompany | **7.7%** |

Vocabularies barely intersect either — one shared department (`Sales`) out of 3 vs 6, and one shared job role (`Sales Executive`) out of 9 vs 13. A different company again.

**`Employee_Performance_Dataset.csv`** (5,000 rows, 6-digit IDs 100021–999957) — **zero** ID intersection with `employee_attrition`, with `hr_performance_engagement`, and with `employee_performance_pro`. It connects to nothing.

Three further checks:
- The two archive files do not link to **each other** — 0 shared names across 500 and 4,863.
- Neither maps to O\*NET: 0 of 13 and 0 of 15 job-role titles match exactly.
- Neither contains employee-level skill data, so **F3 stands unchanged**.

**Verdict: no file in `data/raw/` is replaced.** Substituting either archive file would not repair F1 — it would add a third and fourth unlinked population to the problem. The archive is best read as *confirmation* of the diagnosis: four HR datasets from four different sources, where the O\*NET role/skill ontology is the only genuine connective tissue in the whole collection. That is exactly what the §2 architecture is built on.

*(Note also: `employee_performance_pro.csv` carries direct PII — `Name`, `PhoneNumber`, `CountryCode` — a second reason to keep it out of the project. If a third external validation population is ever wanted for WOW component B, it could serve that role and only that role: scored, never joined, PII stripped. The brief's scope rule — only `enterprise_hr_ai/data/` — keeps it out by default.)*


---

## 2. Corrected data architecture

The five datasets form **two employee populations bridged by one shared role/skill ontology** — not one joined table.

```
  POPULATION A                                  POPULATION B
  employee_attrition.csv (1,470)                hr_performance_engagement.csv (3,000)
  key:   EmployeeNumber                         key:   Employee ID
  label: Attrition          16.1% Yes           label: Voluntarily Terminated   10.8%
  rich features, clean                          engagement/training/survey data
        |                                                     |
        | JobRole (9 distinct)                                | Title (32 distinct)
        v                                                     v
        +--------->  ROLE -> O*NET SOC CROSSWALK  <-----------+
                     semantic + human-reviewed, 41 roles
                                  |
                                  v
                     occupation_data.csv (1,016 SOC codes)
                          |                        |
                          v                        v
              essential_skills.csv          software_skills.csv
              10 foundational skills        8,753 tools / 134 categories
              IM 1-5, LV 1-7                Hot Technology, In Demand
              910 occupations covered       923 occupations covered
```

**Why this is better than the original design, not a compromise:**

- Nothing is fabricated. Every join has verified referential integrity (`essential_skills ⊆ occupation_data` ✓, `software_skills ⊆ occupation_data` ✓).
- Population B carries its **own attrition label** (`Voluntarily Terminated`, 10.8%). That is not a consolation prize — it enables genuine external validation of a model trained on Population A (WOW component B, §7).
- The skills layer serves both populations through one ontology, so the org-wide skill-gap view covers 4,470 employees instead of 1,470.

---

## 3. The one working-practice change: src-first

The Build Notes plan to write logic in notebook cells across Days 1–3, then refactor into modules on Day 4 (Step 17). In production work this is the single most common point of collapse: by Day 4 the notebooks and the modules disagree, and "refactor" quietly becomes "rewrite and re-verify everything."

**Rule for this build — the lab / factory split:**

> `src/hrai/` is the factory. `notebooks/` is the lab.
> Every notebook **imports** from `src/hrai/`. No notebook defines pipeline logic inline.
> A notebook's job is narrative, charts, and evidence — not implementation.

Consequences:
- Step 17 becomes a checkpoint, not a rewrite.
- Every function is unit-testable from the day it is written (Step 22 stops being a retrofit).
- The API (Step 18) and the notebooks call **the same code**, so the dashboard cannot silently disagree with the analysis.
- Re-running a step is `make data` — reproducible, not a manual cell-by-cell ritual.

This does not change *what* gets built, *when*, or *with what*. All 29 steps and their order stand.

---

## 4. Phase 0 — Foundation (before Day 1)

Not in the Build Notes; required for the stated "production and scalability" bar.

| # | Task | Detail |
|---|---|---|
| 0.1 | **Python 3.11 environment** | `conda create -n hrai python=3.11`. Current machine has system Python **3.9.6** (end-of-life, October 2025) with no pandas, plus an Anaconda 3.13 base. 3.9 lacks reliable modern wheels for shap/xgboost; installing into `base` contaminates other work. 3.11 is the stability sweet spot for this stack. |
| 0.2 | **Pin `requirements.txt`** | Currently empty. Exact pins, split into `requirements.txt` / `requirements-dev.txt`. Full library list in §9. |
| 0.3 | **Complete the folder scaffold** | Add `frontend/`, `tests/`, `docs/`, `conf/`, `src/hrai/`, `data/interim/`, `data/predictions/` to match the Build Notes structure. |
| 0.4 | **Engineering guardrails** | `.gitignore` (data, models, `.env`); `pre-commit` running `black`, `ruff`, and **`nbstripout`** — mandatory, so notebook outputs containing employee PII never enter git history; `Makefile`; `pytest.ini`; `.env.example`. |
| 0.5 | **Config-driven design** | `conf/config.yaml` holds every path, threshold, seed, and column contract. Zero hard-coded paths inside `src/`. |
| 0.6 | **Determinism contract** | One `RANDOM_SEED = 42` in config, threaded through splits, models, and the derived-skills generator. Same inputs must always produce byte-identical outputs. |
| 0.7 | **Structured logging + run IDs** | JSON logs with a run ID from the first script, so Steps 20/21 inherit it rather than bolt it on. |

**Gate:** `make setup && make test` passes on a clean clone before Day 1 begins.

---

## 5. The per-step execution cycle

Every one of the 29 steps runs the same seven beats. This is the "step by step execution cycle" the brief asks for.

```
  1. CONTRACT    define/extend the schema + config entry for this step
  2. BUILD       implement in src/hrai/<module>.py  (pure, typed, no I/O surprises)
  3. TEST        pytest unit test written alongside, not after
  4. NARRATE     notebooks/NN_*.ipynb imports the module, shows evidence + charts
  5. PERSIST     write versioned artifact (Parquet/joblib) with a checksum
  6. DOCUMENT    update data dictionary / model card / ADR
  7. GATE        tick the Build Notes checklist, commit with the step number
```

**Definition of Done for a step:** module exists, test passes, notebook runs top-to-bottom on a clean kernel, artifact is on disk with a checksum, docs updated, committed.

**Quality gates between phases:**
- End of Day 1 → all five Pandera schemas pass; `docs/data_relationships.md` signed off; **no employee-level join exists in the codebase.**
- End of Day 2 → model artifact is a single self-contained pipeline; leakage register clean; metrics reproduce from a fresh run.
- End of Day 3 → every role has a reviewed SOC mapping; derived-skill rows carry `is_derived = true`.
- End of Day 4 → `pytest` ≥ 80% coverage on `src/`; API and dashboard read from the same services.

---

## 6. Step-by-step build

### Day 1 — Data Foundation (Steps 01–04)

**01 · Data Understanding** → `notebooks/01_data_understanding.ipynb`, `src/hrai/profiling.py`
Automated profile per dataset: shape, dtypes, missingness, cardinality, candidate keys, constant columns, category vocabularies. Deliverables: `docs/data_dictionary.md` (human) and `conf/schemas/*.yaml` (machine).
*Carries findings F1, F5, F6.*

**02 · Data Validation** → `src/hrai/validation/`
Go straight to **Pandera** rather than the plain-pandas MVP. The Build Notes name Pandera as the destination ("so the rules live in one place instead of scattered across notebooks"); writing throwaway asserts first costs more than it saves. Rules encode the real defects from F6, including the corrected 1–5 engagement range and the `ExitDate` vs `EmployeeStatus` contradiction. Two modes: `--strict` (fail the pipeline) and `--report` (log and continue). Schemas live in `app/validation/` per the Build Notes' target layout and are imported by both the pipeline and the API.

**03 · Data Cleaning** → `src/hrai/cleaning/`
Idempotent pure functions, one module per dataset. Handles: missing values, the 150 duplicate IDs, type coercion, mixed date parsing, category normalisation (whitespace strip, casing), outliers, and skill-name canonicalisation (`AWS` / `Amazon Web Services` / `AWS Cloud` → one token).
**PII:** `FirstName`, `LastName`, `ADEmail`, `Supervisor` dropped; a salted hash retained only if a stable pseudonymous key is needed.
**Format:** Parquet as the working format (typed, compressed, columnar — this is the scalability lever), with a CSV mirror written under the exact filenames the Build Notes specify so the documented structure holds.
Raw files are never modified.

**04 · Data Relationships** → `docs/data_relationships.md`
The ERD from §2, with the evidence table from F1 recorded next to the decision. Every pair of tables: join key, cardinality, referential-integrity check result, and rationale. First **ADR** (`docs/adr/001-two-population-architecture.md`) written here.

### Day 2 — Machine Learning (Steps 05–09)

**05 · Feature Engineering** → `src/hrai/features/`
All preprocessing lives **inside** a scikit-learn `Pipeline` + `ColumnTransformer`. Nothing is fitted outside it. This is what makes the joblib artifact self-contained and makes train/serve skew structurally impossible — the highest-value production practice in the whole build.
- Drop: constants (`EmployeeCount`, `Over18`, `StandardHours`), `EmployeeNumber`.
- Engineered, each with a written rationale per the Build Notes' rule: `IncomePerYearAtCompany`, `PromotionGap`, `SatisfactionIndex`, `ExperienceRatio`, `TenureInRoleRatio`, `ManagerStability`.
- Define the **Common Feature Contract** here — the feature subset shared with Population B, which WOW component B depends on.

**06 · Baseline Model** → Logistic Regression
Per the Build Notes: explainable, fast, real probabilities. Two upgrades:
- **Stratified K-fold CV** reported as mean ± std, not a single split. With only 237 positives, one split is high-variance and would make Step 07's comparison unreliable.
- **Calibration** (Brier score + reliability curve) alongside precision / recall / F1 / ROC-AUC, plus **PR-AUC**. Rationale: Step 16 buckets probabilities into HIGH/MEDIUM/LOW risk bands, so the probabilities must be *correct*, not merely correctly *ranked*. An uncalibrated model ranks fine and bands badly.

**07 · Model Comparison** → LR vs Random Forest vs XGBoost
Identical pipeline and folds for all three. Class imbalance via `class_weight` / `scale_pos_weight`.
The Build Notes say to pick on "the actual cost of mistakes" — made quantitative: a false negative costs a replacement (a configurable multiple of annual `MonthlyIncome`), a false positive costs an unnecessary intervention. We select **model *and* operating threshold** together on expected cost, not by reading the top of a metric column. Winner → `models/attrition_pipeline.joblib`.

**08 · SHAP Explainability**
`TreeExplainer` for tree models, `LinearExplainer` for LR. Global summary + per-employee local force plots.
**Production addition:** precompute and cache SHAP background values into `models/vN/shap_background.joblib`. Computing SHAP inside a request would make the "why is this person flagged" endpoint unusably slow.

**09 · Model Versioning + MLflow**
The Build Notes' `models/vN/metadata.json` scheme first, then MLflow on a local file store — in that order, as instructed. Metadata is expanded to what an audit actually needs: git SHA, input data checksums, feature list, metrics with CI, chosen threshold, seed, library versions, training date. Plus `docs/model_card.md` (intended use, limitations, fairness results, out-of-scope uses).

### Day 3 — Workforce Intelligence (Steps 10–16)

**10 · Engagement Analytics**
Grain resolved first (F5). Aggregations by `DepartmentType`, `Division`, `Title`, tenure band. Correct 1–5 scale. Lowest-engagement cohort surfaced for HR. No ML, per the Build Notes.

**11 · Role Intelligence + the Crosswalk** ← *keystone step*
`occupation_data.csv` becomes the role master. Then the crosswalk that F4 makes mandatory:
1. Embed each of the 41 role titles (9 + 32) and all 1,016 O\*NET title+description pairs with `sentence-transformers/all-MiniLM-L6-v2`.
2. Cosine similarity → top-5 candidates per role with confidence scores.
3. **Human review** of all 41 into `conf/role_crosswalk_reviewed.yaml`. 41 is small enough to get right by hand; leaving it purely automatic is where a plausible-looking wrong answer would enter the system.
4. Low-confidence mappings flagged and surfaced in the UI rather than hidden.

**12 · Employee Skills Table**
The derived proficiency table (F3), built only from real signals:
- role's O\*NET requirement via the crosswalk sets the expected profile;
- `Training Program Name` + `Training Outcome` (Passed/Completed raise proficiency, Failed/Incomplete do not);
- `Current Employee Rating`, `Performance Score`, tenure, `Education`, `EducationField`, `JobLevel`, `TrainingTimesLastYear` modulate it.
Deterministic under the seed. Every row carries `is_derived = true`, documented in the data dictionary and surfaced as a banner in the dashboard. **We never present derived proficiency as observed fact.**

**13 · Skill Gap Engine**
Two tiers, per F2:
- *Foundational:* required O\*NET `LV` vs employee proficiency → graded gap, weighted by `IM` importance.
- *Technical:* set subtraction over the role's tool set, weighted by `Hot Technology` and `In Demand`.
Severity = importance × gap magnitude. Core logic stays the plain set operation the Build Notes specify.

**14 · Organisation-Wide Skill Gap**
Rollup across both populations. Severity thresholds converted from the Build Notes' absolute counts (100+/50+) to **percentage-of-population** bands, so they stay meaningful as headcount changes — the same rule, made scalable.

**15 · Recommendation Engine**
v1 rule-based, exactly as specified. v2 semantic: sentence-transformers + cosine similarity between the missing skill and course descriptions, so `MLOps` matches *"Deploying and Monitoring Machine Learning Systems"*.
Course catalogue is built from in-data sources only: the 5 real `Training Program Name` values, plus O\*NET skill and tool names, with real `Training Cost` and `Training Duration(Days)` distributions attached — which is what makes the ROI engine in §7 possible.

**16 · Employee Intelligence Table**
The unified business output, one row per employee, with a `population` column. Attrition probability is native for Population A and transferred for Population B (§7B). Written to Parquet; this table *is* the dashboard's data source.

### Day 4 — Application (Steps 17–23)

**17 · Refactor into modules** — a checkpoint under the src-first rule (§3). Verify the Build Notes' `app/` layout (`api/`, `services/`, `validation/`, `ml/`, `utils/`) is fully populated and no logic remains stranded in notebooks.

**18 · FastAPI backend** — the six specified endpoints:
`POST /predict/attrition` · `GET /dashboard/summary` · `GET /dashboard/attrition-by-department` · `GET /dashboard/skill-gaps` · `GET /dashboard/recommendations` · `GET /employees/{employee_id}`
Production shape: app-factory pattern, routers per domain, **model loaded once in the lifespan handler** (never per request — the difference between ~5 ms and ~800 ms responses), dependency-injected services, typed response models, `/health` + `/ready`, `/api/v1` prefix, auto-generated OpenAPI docs.

**19 · API input validation** — Pydantic v2 models sharing the column contract with the Pandera schemas, so validation rules have exactly one definition. Bad input gets a 422/400 and never reaches the model.

**20 · Logging** — structured JSON, request-correlated, covering the lifecycle the Build Notes list (startup, dataset loaded, prediction requested, model version, completion, errors).

**21 · Prediction logging** — append-only Parquet under `data/predictions/`, partitioned by date: timestamp, employee ID, model version, feature hash, probability, threshold, risk band. This is the feed that Step 25's drift monitoring consumes.

**22 · Unit testing** — pytest, the six cases the Build Notes list, plus schema contract tests, a golden-file test for skill-gap output, and API tests via `TestClient`. Coverage target ≥ 80% on `src/`.

**23 · Streamlit dashboard** — the specified layout: KPI cards, department filter, risk distribution chart, skill-gap chart, recommendation table, employee drill-down. The frontend calls the **API**, never the services directly, so the separation stays honest and the API stays the contract. Plus the Retention Planner page (§7).

### Later — Enterprise Hardening (Steps 24–29)

**24 Docker** — multi-stage builds, non-root user, separate backend/frontend containers, `docker-compose.yml`, healthchecks, `.dockerignore`.
**25 Drift monitoring** — pandas baseline (PSI + Kolmogorov–Smirnov) on the Build Notes' watch-list features and the prediction distribution, then **Evidently AI** reports once the basic version proves useful, in that order.
**26 Model performance monitoring** — ground-truth join, metric recomputation on live data.
**27 Retraining strategy** — the Build Notes' rule (drift > threshold OR F1 below threshold OR 6 months of new data), written as a runbook with an approval gate before deploy.
**28 Documentation** — full README per the Build Notes' 12-point list, plus model card, data dictionary, relationships doc, and the ADR set capturing findings F1–F8.
**29 Deployment** — the Build Notes' target architecture, with CI (lint → test → build) on GitHub Actions.

---

## 7. The WOW: Retention ROI Copilot

Every capable version of this project answers *"who is likely to leave, and why?"* That is where the Build Notes stop, and it is where the field stops. **It is not the question an HR director actually has.** Theirs is: *"I have a retention budget. Who do I spend it on, on what, and what do I get back?"*

The Retention ROI Copilot answers that — and it is only buildable because the rest of this plan exists, since it consumes the model, SHAP, the skill gaps, the recommendations, and the real training-cost data at once.

### A · Semantic Skill Graph *(infrastructure, and impressive on its own)*
The reviewed Role→O\*NET crosswalk (Step 11) turns three inert reference files into a live, queryable skill ontology spanning 1,016 occupations, 10 graded foundational competencies, and 8,753 technical tools — with `Hot Technology` and `In Demand` market signals attached. This is what lets the platform say *"this role needs these skills at this level, and these ones are heating up in the market."*

### B · Cross-population transfer validation *(credibility)*
Train on Population A. Score Population B through the Common Feature Contract. **Validate against Population B's own `Voluntarily Terminated` label** (10.8% base rate vs A's 16.1%).

This turns finding F1 from a defect into a genuine ML result: a measured, honestly reported answer to *"does this model survive contact with a different workforce?"* — external validation that almost no project at this level attempts. It doubles as the ground truth for the drift work in Step 25.

### C · Counterfactual Action Engine *(the headline)*
For each at-risk employee, perturb only **actionable** levers through the fitted pipeline and measure the change in predicted risk:

| Lever | Perturbation | Cost source |
|---|---|---|
| Reduce overtime | `OverTime`: Yes → No | backfill / capacity estimate |
| Work-life balance | `WorkLifeBalance` +1 | programme cost |
| Targeted training | `TrainingTimesLastYear` +1 | **real `Training Cost` from the data** |
| Promotion | `YearsSinceLastPromotion` → 0, `JobLevel` +1 | banded from `MonthlyIncome` |
| Compensation | `MonthlyIncome` +x% | direct |
| Equity | `StockOptionLevel` +1 | direct |

Then:
1. **Δrisk per lever**, and for the best two-lever combination.
2. **Expected value saved** = Δrisk × replacement cost (a configurable multiple of annual salary).
3. **ROI ranking** = value saved ÷ intervention cost.
4. **Budget-constrained org plan** — a knapsack over all employees: *"with a ₹X budget, these are the N interventions, on these N people, that maximise expected retained value."*
5. **Fused with the skills half** — when the winning lever is training, the specific course comes from the Step 15 recommender against that person's actual skill gap. The two halves of the project become one decision.

New endpoints: `POST /intelligence/counterfactual/{employee_id}` · `GET /intelligence/action-plan?budget=X`
New dashboard page: **Retention Planner** — budget slider, ranked action list, projected risk reduction, total expected value retained.

### D · Fairness audit *(non-negotiable for HR AI)*
An HR model that influences decisions about people needs a bias check, and this one is cheap to do well: group metrics (TPR / FPR / calibration) sliced by `Gender`, age band, and `MaritalStatus`; equal-opportunity difference reported in the model card. **Protected attributes are excluded from the counterfactual levers by design** — the system will never propose an intervention on someone's age, gender, or marital status.

### Honesty guardrails
The counterfactual engine is **association-based decision support, not causal inference**. Every output states this. Counterfactuals assume feature independence, which is imperfect. SHAP attributions are shown alongside every recommendation, and a human approves before anything is acted on. These caveats ship *in the UI*, not buried in a README — an HR tool that overstates its certainty is a liability, and saying so is part of the deliverable.

---

## 8. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Fabricated employee join corrupts everything downstream | F1; two-population architecture; **CI check that fails the build if an employee-level join between A and B appears in the code** |
| R2 | Derived skills mistaken for observed fact | `is_derived` flag on every row; data dictionary; dashboard banner; documented derivation contract |
| R3 | Target leakage → perfect-looking, worthless model | Leakage register at schema level (F7); all preprocessing inside the pipeline object |
| R4 | Only 237 positive cases → unstable metrics | Stratified K-fold CV, mean ± std, calibration checks, confidence intervals on reported metrics |
| R5 | Notebook / module drift by Day 4 | src-first rule (§3); notebooks import, never implement |
| R6 | Employee PII leaking into git history | `nbstripout` in pre-commit (mandatory); PII columns dropped in cleaning; `data/` gitignored |
| R7 | Wrong O\*NET role mappings silently distort all skill gaps | Human review of all 41 mappings; confidence scores persisted and surfaced |
| R8 | Python 3.9.6 (EOL) breaks on modern ML wheels | Dedicated conda env on 3.11 |
| R9 | Counterfactual advice read as causal guarantee | Explicit in-UI caveats; SHAP shown alongside; human-in-the-loop approval; protected attributes excluded as levers |

---

## 9. Toolchain

Exactly as specified in the Build Notes, pinned.

| Layer | Libraries |
|---|---|
| Core | `pandas`, `numpy`, `pyarrow`, `scipy` |
| Validation | `pandera`, `pydantic` v2 |
| ML | `scikit-learn`, `xgboost`, `shap`, `joblib` |
| Tracking | `mlflow` |
| NLP | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| API | `fastapi`, `uvicorn` |
| Frontend | `streamlit`, `plotly` (dashboard), `matplotlib` (notebooks) |
| Monitoring | `evidently` |
| Testing | `pytest`, `pytest-cov`, `httpx` |
| Quality | `black`, `ruff`, `pre-commit`, `nbstripout` |
| Config | `pyyaml`, `python-dotenv` |
| Packaging | Docker, docker-compose |

---

## 10. Sequencing

The Build Notes' four-day framing is kept as the phase structure. Calendar days are given as working-session estimates at a production-quality bar — the gates matter more than the clock.

| Phase | Steps | Sessions | Exit gate |
|---|---|---|---|
| **0 · Foundation** | 0.1–0.7 | 0.5 | `make setup && make test` green on a clean clone |
| **1 · Data Foundation** | 01–04 | 1.5 | Schemas pass; relationships signed off; no A↔B employee join exists |
| **2 · Machine Learning** | 05–09 | 1.5 | Self-contained pipeline artifact; leakage clean; metrics reproduce |
| **3 · Workforce Intelligence** | 10–16 | 2 | All 41 roles mapped and reviewed; intelligence table built |
| **4 · Application** | 17–23 | 2 | API + dashboard live; coverage ≥ 80% |
| **5 · WOW layer** | A–D | 1.5 | Action plan endpoint + Retention Planner page + fairness audit in model card |
| **6 · Hardening** | 24–29 | 1.5 | `docker compose up` works from a clean clone; drift + retraining runbook written |

The Build Notes' governing rule stands and is adopted verbatim: **do not jump ahead to SHAP, MLflow, Docker, or deployment before the four Day-1 steps are actually finished.**

---

## 11. Decisions needed before build starts

1. **WOW scope** — full Copilot (A+B+C+D), or the Skill Graph and cross-population validation only (A+B)?
2. **Currency and replacement-cost multiple** for the ROI model (default assumption: ₹, replacement cost = 6× monthly salary).
3. **Confirm the F1 correction** — this plan does not join the two employee datasets. That is a deliberate departure from Step 04 of the Build Notes, on the evidence in §1.
