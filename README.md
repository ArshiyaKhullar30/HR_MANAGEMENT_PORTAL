<div align="center">

# Enterprise HR AI
### Workforce Intelligence & Upskilling Platform

**Who is at risk of leaving · where the skill gaps are · what to actually do about it**

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-146%20passing-3E7A5E)](#12--testing)
[![Coverage](https://img.shields.io/badge/coverage-90%25-3E7A5E)](#12--testing)
[![Model](https://img.shields.io/badge/ROC--AUC-0.81-2F5D8A)](#8--the-model)

</div>

---

## Read this part first

Most HR AI projects answer *"who is likely to leave?"* This one does too — but that
is the easy half, and it is not the question an HR director actually has.

Theirs is: **"I have a retention budget. Who do I spend it on, on what, and what do
I get back?"**

That question is what this platform is built around. Everything else — the data
cleaning, the model, the skill ontology — exists to make that answer trustworthy.

And a warning that shapes the entire codebase: **before writing a line of pipeline
code, we tested the assumption the project was specified on, and it was wrong.**
Section 3 explains. It is the most important thing in this repository.

---

## Table of contents

| | |
|---|---|
| [1 · What this does](#1--what-this-does) | [8 · The model](#8--the-model) |
| [2 · Quick start](#2--quick-start) | [9 · The Retention ROI Copilot](#9--the-retention-roi-copilot) |
| [3 · The finding that changed everything](#3--the-finding-that-changed-everything) | [10 · API reference](#10--api-reference) |
| [4 · Architecture](#4--architecture) | [11 · Dashboard](#11--dashboard) |
| [5 · The execution flow](#5--the-execution-flow) | [12 · Testing](#12--testing) |
| [6 · Repository map](#6--repository-map) | [13 · Monitoring & retraining](#13--monitoring--retraining) |
| [7 · The data](#7--the-data) | [14 · What we would do next](#14--what-we-would-do-next) |

---

## 1 · What this does

Five raw CSVs go in. Four things come out.

| Capability | What it answers | How |
|---|---|---|
| **Attrition risk** | Who is likely to leave, and *why* | Calibrated Logistic Regression + SHAP |
| **Engagement analytics** | Where morale is failing | Aggregation on 3,000 employees |
| **Skill gaps** | What the organisation is missing | O\*NET two-tier ontology, set subtraction |
| **Retention ROI Copilot** | **What to do, for whom, at what cost** | Counterfactuals + budget knapsack |

The first three are the brief. The fourth is the reason the platform is worth
running.

---

## 2 · Quick start

You need [conda](https://docs.conda.io/en/latest/miniconda.html). About 8 minutes
end to end, most of it downloading PyTorch.

```bash
git clone <this-repo> && cd enterprise_hr_ai

# 1 — environment (Python 3.11; 3.9 is end-of-life and lacks modern ML wheels)
conda create -y -n hrai python=3.11
conda activate hrai
conda install -y -c conda-forge llvm-openmp     # macOS only: XGBoost needs OpenMP
make setup                                       # deps + git hooks

# 2 — build everything from the raw CSVs
make pipeline        # Steps 01-04: profile -> validate -> clean
make train           # Steps 05-09: features -> compare 3 models -> calibrate -> version
make intelligence    # Steps 10-16: skills, gaps, recommendations, the unified table

# 3 — run it
make api             # http://localhost:8000/docs
make dashboard       # http://localhost:8501     (in a second terminal)
```

Or with Docker, which skips all of the above:

```bash
docker compose up --build
```

**Sanity check:** `make test` should print `146 passed` and `coverage 90%`.

**Want to verify it properly?** [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) walks
through it in eight levels, from a 30-second health check to deleting every artifact
and proving it rebuilds byte-identically.

<details>
<summary><b>Every make target</b></summary>

| Target | Does |
|---|---|
| `make setup` | Install dependencies and git hooks |
| `make pipeline` | Steps 01–04 — profile, validate, clean |
| `make train` | Steps 05–09 — feature engineering through model versioning |
| `make intelligence` | Steps 10–16 — the workforce intelligence layer |
| `make api` | Serve the FastAPI backend |
| `make dashboard` | Serve the Streamlit dashboard |
| `make test` | Full suite with the 80% coverage gate |
| `make lint` / `make format` | Ruff + Black |
| `make check-invariant` | Fail if anyone joined the two employee populations |
| `make lock` | Freeze the resolved environment |

</details>

---

## 3 · The finding that changed everything

The project brief specified this, in `HR_AI_Project_Build_Notes.docx`, Step 04:

> *"employee_attrition joins to hr_performance_engagement on EmployeeID,
> one-to-one, because it's the same employee's performance record."*

The two files share **753 employee IDs**. That looks like a join waiting to happen.

Before building on it, we checked whether those 753 IDs describe the same people.

| Check | Result | What it should be |
|---|---|---|
| Shared IDs | 753 | — |
| **Gender agrees** | **48.6%** | ~100% |
| **Age agrees (±1 year)** | **6.0%** | ~100% |

```
ID     employee_attrition says        hr_performance_engagement says
1001   27, Female, Lab Technician     born 1957 (66), Female, Software Engineer
1002   45, Male,   Lab Technician     born 1950 (73), Female, Software Engineer
1003   47, Female, Sales Executive    born 1973 (50), Female, Software Engineer
```

Gender agreeing 48.6% of the time is a coin flip. **These are two different
companies whose ID ranges happen to overlap.** Performing that join would have
invented 753 employees who do not exist and quietly corrupted every downstream
number — risk scores, engagement, skill gaps, recommendations, all of it.

> The build notes had already warned about exactly this:
> *"Don't merge anything yet, even if two files look like they share an ID.
> Confirm the IDs actually refer to the same employees before joining — a matching
> column name isn't proof of a matching key."*
>
> We confirmed. They don't. **The plan follows that rule rather than the later
> assumption.**

We also tested both spare datasets in `archive/` as replacements. Neither is
related either: `employee_performance_pro.csv` agrees on 1.3% of ages, and
`Employee_Performance_Dataset.csv` shares **zero** IDs with anything.

### How the codebase enforces it

This is not a note in a document. It is three mechanisms:

1. **`scripts/check_no_cross_population_join.py`** runs in pre-commit and CI. It
   fails the build if any `merge`/`join` is keyed on an employee ID in a file that
   touches both populations. It is precise enough not to cry wolf — `person_key`
   joins, vertical `concat` and `str.join` all pass.
2. **`person_key`** (`"A-101"`, `"B-1500"`) is the only identity used downstream.
   A bare numeric ID is never a person here.
3. **A live API contract test.** `GET /employees/resolve/1001` must return *two*
   people, not one:

```json
{"matches": [
  {"person_key": "A-1001", "role": "Laboratory Technician", "department": "Research & Development"},
  {"person_key": "B-1001", "role": "Software Engineer",     "department": "Software Engineering"}
]}
```

### The unexpected payoff

Population B carries its **own** attrition label (`Voluntarily Terminated`, 10.8%).
So instead of *assuming* the model generalises, we measured it — see
[section 9](#b--cross-population-validation). Almost no project at this scale gets
a genuine external validation set. This constraint handed us one.

---

## 4 · Architecture

The five datasets are two employee populations bridged by **one shared skill
ontology** — not one joined table.

```
  POPULATION A                                  POPULATION B
  employee_attrition.csv (1,470)                hr_performance_engagement.csv (3,000)
  key:   EmployeeNumber                         key:   Employee ID
  label: Attrition          16.1%               label: Voluntarily Terminated  10.8%
        │                                                     │
        │      ✗────── NO EMPLOYEE-LEVEL JOIN ──────✗         │
        │                                                     │
        │ JobRole (9)                              Title (31) │
        ▼                                                     ▼
        └──────────►  ROLE → O*NET SOC CROSSWALK  ◄───────────┘
                     semantic + human-reviewed · 40 roles
                                  │
                                  ▼
                     occupation_master (1,016 SOC codes)
                          │                        │
                          ▼                        ▼
              essential_skills              software_skills
              TIER 1 · FOUNDATIONAL         TIER 2 · TECHNICAL
              10 skills, graded 1-7         8,751 tools, binary
              910 occupations               923 occupations
```

Because the bridge is the ontology rather than employee identity, the skill-gap
view covers **4,470 employees** instead of 1,470.

### Runtime

```
                          HR USER
                             │
                   Streamlit dashboard  :8501
                             │  HTTP (never imports services directly)
                    FastAPI backend     :8000
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ML prediction        Skill engine          Analytics
  + SHAP + ROI         + crosswalk           + engagement
        └────────────────────┼────────────────────┘
                             ▼
                Employee Intelligence Table  (Parquet)
                             │
                 Prediction log → drift monitoring
                             │
                    Model registry + MLflow
```

---

## 5 · The execution flow

All 29 steps of the brief, in order. Each runs the same seven beats:

```
  1 CONTRACT  →  2 BUILD  →  3 TEST  →  4 NARRATE  →  5 PERSIST  →  6 DOCUMENT  →  7 GATE
    schema       src/         pytest     notebook      checksum      docs/          commit
```

### Day 1 — Data foundation · `make pipeline`

| Step | What happens | Output |
|:--:|---|---|
| **01** | Profile all five datasets. Reproduce findings F1, F4, F5, F6 from code, not memory | `docs/data_dictionary.md` |
| **02** | Pandera contracts. Raw data fails **1,198 checks** — that is the point | `docs/validation_report.md` |
| **03** | Clean, drop PII, resolve grain, canonicalise skill names. Then re-validate **strict** | `data/processed/*.parquet` |
| **04** | Document every table pair with its integrity result | `docs/data_relationships.md` |

> **Gate:** all five schemas pass, and no employee-level join exists in the codebase.

<details>
<summary><b>What Step 03 actually fixes</b></summary>

| Defect | Count | Fix |
|---|--:|---|
| `ExitDate` set but status is Active / Future Start / Leave | 1,198 | `EmployeeStatus` ruled authoritative — see below |
| Whitespace-padded category (`'Production       '`) | 2,115 | Stripped |
| Mixed date formats (`20-Sep-19` vs `07-10-1969`) | all date cols | Multi-format parser |
| Constant columns (`EmployeeCount`, `Over18`, `StandardHours`) | 3 cols | Dropped |
| Direct PII (`FirstName`, `LastName`, `ADEmail`, `Supervisor`) | 4 cols | Dropped outright |
| Duplicate employee rows | 150 | Collapsed to employee grain |

**Which column was telling the truth?** Cross-tabulating settled it:
`TerminationType` is distributed almost uniformly *within every* `EmployeeStatus`
— `Voluntarily Terminated` employees split 91/77/81/90 across
Involuntary/Resignation/Retirement/Voluntary. That is noise. And `ExitDate` is
populated even for `Future Start` employees, who cannot have left. So
`EmployeeStatus` is the label, and the other three are dropped as unreliable —
which also satisfies the leakage register.

</details>

### Day 2 — Machine learning · `make train`

| Step | What happens |
|:--:|---|
| **05** | Six engineered features, each with a written reason. **All preprocessing lives inside the pipeline object** |
| **06** | Logistic Regression baseline. Repeated stratified 5-fold × 3 — one split is too noisy with 237 positives |
| **07** | LR vs Random Forest vs XGBoost, identical folds. Threshold chosen by **expected cost**, not F1 |
| **08** | SHAP: global drivers and per-employee reasons. Background cached per model version |
| **09** | `models/vN/` + `metadata.json`, then MLflow — in that order, as the brief instructs |

> **Gate:** the artifact is a single self-contained pipeline; leakage register clean;
> metrics reproduce from a fresh run.

### Day 3 — Workforce intelligence · `make intelligence`

| Step | What happens |
|:--:|---|
| **10** | Engagement analytics on the **correct 1–5 scale** and the **correct grain** |
| **11** | 🔑 **The role → O\*NET crosswalk.** Semantic match, then all 40 reviewed by hand |
| **12** | Derived employee skills — deterministic, seeded, labelled `is_derived` everywhere |
| **13** | Skill gap engine: set subtraction for tools, graded comparison for competencies |
| **14** | Org-wide rollup with percentage-based severity bands |
| **15** | Recommendations: rules first, sentence-transformers second |
| **16** | The unified Employee Intelligence Table — the business output |

> **Gate:** all 40 roles reviewed; every derived row flagged.

### Day 4 — Application · `make api` + `make dashboard`

| Step | What happens |
|:--:|---|
| **17** | Refactor — a *checkpoint*, not a rewrite (see [src-first](#src-first-the-one-practice-change)) |
| **18** | FastAPI: app factory, routers, model loaded **once** in the lifespan handler |
| **19** | Pydantic v2 sharing bounds with the Pandera schemas — one definition per rule |
| **20** | Structured JSON logs, request-correlated |
| **21** | Prediction log → date-partitioned Parquet → the drift feed |
| **22** | pytest, 146 tests, 90% coverage |
| **23** | Streamlit: 7 tabs, talking to the API and never to the services |

### Later — Hardening

Steps **24–29**: Docker, drift monitoring (PSI + KS, then Evidently), performance
monitoring, the retraining rule, documentation, deployment.

### src-first: the one practice change

The brief writes pipeline logic in notebooks across Days 1–3, then refactors into
modules on Day 4. In production work that is the single most common point of
collapse — by Day 4 the notebooks and modules disagree, and "refactor" quietly
becomes "rewrite and re-verify everything," under time pressure.

> **`src/hrai/` is the factory. `notebooks/` is the lab.**
> Every notebook *imports* from `src/`. None of them define pipeline logic.

Step 17 becomes a formality. Every function is testable the day it is written. The
API and the notebooks call the same code, so the dashboard cannot silently
disagree with the analysis. Nothing about *what* is built, *when*, or *with which
libraries* changes.

---

## 6 · Repository map

```
enterprise_hr_ai/
├── conf/
│   ├── config.yaml                    ← every path, threshold, seed, contract
│   ├── role_crosswalk_reviewed.yaml   ← 40 hand-reviewed mappings, each with a reason
│   └── schemas/                       ← machine-readable dataset profiles
├── data/
│   ├── raw/          the 5 source CSVs, never modified
│   ├── processed/    Parquet + CSV mirror + checksum manifest
│   └── predictions/  append-only, date-partitioned — the drift feed
├── src/hrai/                          ← THE FACTORY
│   ├── profiling/    Step 01, 04
│   ├── validation/   Step 02  (Pandera schemas + runner)
│   ├── cleaning/     Step 03
│   ├── features/     Step 05  (engineering + pipeline construction)
│   ├── ml/           Steps 06-09 + transfer, fairness, explainability
│   ├── skills/       Steps 11-15 (crosswalk, ontology, gaps, recommendations)
│   ├── intelligence/ Steps 10, 16 + the counterfactual engine
│   ├── monitoring/   Steps 25-27
│   └── utils/        config, structured logging, checksummed I/O
├── app/                               ← THE SERVICE
│   ├── main.py       app factory + lifespan + middleware
│   ├── api/          attrition · dashboard · skills · intelligence
│   ├── services/     intelligence read model, prediction log
│   ├── ml/           model loader (loads once) + predictor
│   └── validation/   Pydantic request/response models
├── frontend/Home.py                   ← THE DASHBOARD
├── notebooks/        17 numbered notebooks — narrative, not implementation
├── tests/            146 tests, 90% coverage
├── models/vN/        artifact + metadata + SHAP background + base model
├── docs/             data dictionary, ADRs, model card, all generated reports
└── scripts/check_no_cross_population_join.py   ← the invariant guard
```

---

## 7 · The data

| File | Rows | What it is | What we found |
|---|--:|---|---|
| `employee_attrition.csv` | 1,470 | IBM-style HR records, **Population A** | Clean: 0 missing, 0 duplicates. 16.1% attrition |
| `hr_performance_engagement.csv` | 3,150 | Surveys + training, **Population B** | **Event grain**, not employee grain. 3,000 people |
| `occupation_data.csv` | 1,016 | O\*NET occupations | **0 exact title matches** to our roles |
| `essential_skills.csv` | 18,200 | O\*NET **Basic Skills** | Only **10** cognitive skills, graded IM/LV |
| `software_skills.csv` | 31,821 | O\*NET **Technology Skills** | **8,751 tools**, Hot Technology flags |

### Two things the brief got wrong about this data

**`essential_skills.csv` is not technical skills.** It is O\*NET's *Basic Skills*
file: ten cognitive competencies — Reading Comprehension, Critical Thinking,
Mathematics — each scored per occupation on Importance (1–5) and Level (1–7). The
brief's worked example (`{Python, SQL, MLOps, Docker, AWS}`) can only come from
`software_skills.csv`. So the ontology is **two-tier**, which is richer than the
single-tier design, not poorer:

* **Tier 1 · Foundational** — graded. Nobody has *zero* Critical Thinking; the
  question is whether they have enough for the role.
* **Tier 2 · Technical** — binary. You either work with Kubernetes or you do not.

The gap engine does a graded comparison for Tier 1 and genuine set subtraction for
Tier 2. Treating them alike would either lose the grading or invent one.

**Engagement is 1–5, not 0–100.** The brief specifies a 0–100 range check.
Applied to this file it would pass every single row and catch nothing — the most
dangerous kind of validation, because it looks like coverage.

### The keystone: the role → O\*NET crosswalk

Zero of the 40 role titles match an O\*NET occupation exactly. Without a crosswalk,
all three O\*NET files are inert and the entire skills half of the project has no
input.

A sentence-transformer matches them semantically — but the automatic result is not
trustworthy on its own:

| Role | Machine said | Confidence | Reviewed to |
|---|---|--:|---|
| **CIO** | *Editors* | 0.21 | Computer and Information Systems Managers |
| **Senior BI Developer** | *Editors* | 0.36 | Business Intelligence Analysts |
| **BI Director** | *Media Technical Directors* | 0.35 | Computer and Information Systems Managers |
| **Laboratory Technician** | *Dental Laboratory Technicians* | 0.69 | Medical and Clinical Laboratory Technicians |
| **Shared Services Manager** | *Personal Service Managers* | 0.50 | General and Operations Managers |
| Database Administrator | Database Administrators | 0.78 | ✓ confirmed |

Three-letter acronyms are invisible to an embedding model. All **40 mappings were
reviewed by hand**, each with a written reason, in
`conf/role_crosswalk_reviewed.yaml`. All 40 land on occupations with complete
skill data.

### One thing the data does not contain

**No dataset records what skills an individual holds.** The brief anticipated this
and pre-authorised the fallback. So employee skill profiles are *derived* — under
three rules that make them defensible rather than decorative:

1. **Derived, never random.** Driven by real signals: the role's O\*NET
   requirement, tenure, performance rating, education, and actual training history
   *including whether the training was passed or failed*.
2. **Deterministic.** Individual variation comes from a hash of
   `(seed, employee_id, skill)`, not an RNG. Byte-identical every run, on any
   machine, regardless of row order. There is a test for this.
3. **Labelled everywhere.** `is_derived = true` on every row, surfaced in every
   API response and as a banner on the dashboard. Derived proficiency is never
   presented as observed fact.

---

## 8 · The model

**Logistic Regression won.** The brief expected XGBoost ("usually strongest on
tabular data like this"). On *this* data it was not — and reporting the measured
result rather than the expected one is the whole point of running the comparison.

| Model | PR-AUC | Recall | ROC-AUC | Note |
|---|--:|--:|--:|---|
| **Logistic Regression** | **0.637** | 0.735 | 0.832 | ← winner |
| XGBoost | 0.563 | 0.416 | 0.792 | with `scale_pos_weight` for a fair fight |
| Random Forest | 0.549 | 0.239 | 0.797 | |

1,470 rows with a largely linear signal is exactly the regime where a regularised
linear model beats boosting.

### Held-out test performance

| Metric | Value | |
|---|--:|---|
| ROC-AUC | 0.809 | |
| PR-AUC | 0.534 | the honest headline under imbalance |
| Recall | **0.894** | we would rather over-flag than miss a leaver |
| Precision | 0.261 | the deliberate trade — see below |
| F1 | 0.404 | |
| Brier | 0.104 | probabilities are trustworthy, not just ranked |

**Accuracy is not reported anywhere.** At a 16.1% positive rate, predicting "stays"
for everyone scores 83.9% while being useless. Accuracy rewards exactly the
behaviour we are avoiding, so it is excluded from the metric set entirely.

<details>
<summary><b>Why precision is 0.26 — and why that is correct</b></summary>

The brief says to choose the threshold on "the actual cost of mistakes". We made
that quantitative:

* A **false negative** costs a replacement ≈ **6× monthly salary**
* A **false positive** costs one unnecessary retention conversation ≈ **0.25×
  monthly salary**

Both scale with the employee's own salary, so the ratio is constant at **24:1**
and the cost-optimal threshold is necessarily low. The empirical grid search
landed at **0.08**; the closed-form Bayes optimum is **0.04**. They agree, which
is a useful check that the probabilities are genuinely calibrated.

At that threshold the model saves **₹2.6M** in expected cost versus operating at
0.5.

Low precision is the correct answer to the economics — and a useless work queue.
So there is a **second, capacity-bounded operating point** for the dashboard's
watchlist: flag only as many people as HR can realistically review. Cost
optimality and operational reality are different questions and get different
answers.

</details>

<details>
<summary><b>Why sigmoid calibration, chosen on evidence</b></summary>

Both methods were evaluated on out-of-fold predictions:

| Method | Brier | Saturated | Distinct values |
|---|--:|--:|--:|
| **sigmoid** | 0.0961 | **0.0%** | **1,172** |
| isotonic | 0.0930 | 3.6% | 614 |

Isotonic scored marginally better on Brier — and it is a *step function*, so a
band of raw scores all map to exactly 1.0. That is fatal for the counterfactual
engine, which measures risk as a *difference* after nudging one feature: a
saturated probability cannot respond to a perturbation at all. Sigmoid was
selected for smoothness, and the comparison is recorded in the model metadata.

</details>

### Why employees leave — SHAP

| Driver | Importance | Direction |
|---|--:|---|
| JobRole | 0.94 | varies by category |
| OverTime | 0.73 | varies by category |
| TenureInRoleRatio | 0.54 | higher reduces risk |
| YearsAtCompany | 0.52 | higher reduces risk |
| YearsInCurrentRole | 0.47 | higher increases risk |

Direction comes from the correlation between a feature's value and its SHAP value
— not from mean signed SHAP, which is ~0 for every feature by construction and
would produce confident, meaningless directions. Categorical features are labelled
"varies by category", because a category has no *higher value*.

Per-employee explanations sum one-hot columns back onto their source feature, so a
reader sees **OverTime** once rather than "OverTime: Yes" and "OverTime: No"
competing as two drivers:

```
Employee A-1  (actually left)
   OverTime            = Yes       shap=+1.1121  increases risk
   NumCompaniesWorked  = 8         shap=+1.0999  increases risk
   WorkLifeBalance     = 1         shap=+0.6047  increases risk
```

### Fairness audit

| Attribute | Equal-opportunity diff | Verdict |
|---|--:|---|
| Gender | 0.032 | ✅ within tolerance |
| MaritalStatus | 0.101 | ⚠️ flagged |
| Age band | 0.206 | ⚠️ flagged |

The flags are real, and the interpretation matters: **every group's calibration gap
is under 0.04.** The model is well-calibrated *within* each group; the disparity in
flag rates follows from genuinely different base rates (18–29 leave at 27.9%, 50+ at
13.3%). Calibration and equalised odds cannot both hold when base rates differ —
that is a theorem, not a bug. So the honest report says which is being sacrificed,
routes flagged groups to human review, and never uses the score as an automatic
decision.

**Protected attributes can never be intervention levers.** Age, gender and marital
status are excluded structurally, and a test asserts the two sets never intersect.

---

## 9 · The Retention ROI Copilot

### A · Counterfactual interventions

For each at-risk employee, perturb one **actionable** lever through the fitted
pipeline and measure the change in predicted risk.

```
Employee A-622   baseline risk 93.5% (HIGH)   replacement cost 14,040

INTERVENTION                 CHANGE          NEW RISK   DROP     COST     VALUE    ROI
Remove mandatory overtime    Yes → No           81.3%   12.2%    2,808    1,718   0.61
Work-life balance programme  1 → 2              92.0%    1.6%    1,170      218   0.19
Targeted upskilling          3 → 4              92.9%    0.7%      576       92   0.16
Additional stock options     0 → 1              92.4%    1.1%    1,404      159   0.11
Compensation increase        2340 → 2574        93.4%    0.1%    2,808       17   0.01

BEST PAIR: Remove overtime + Targeted upskilling → 93.5% to 79.6%, cost 3,384
```

Costs scale with the employee's own salary, so retaining a senior engineer costs
more than retaining a junior one — matching how the benefit is calculated.
**Training is priced from the real `Training Cost` distribution in the engagement
data**, not from an invented number.

When the winning lever is training, the specific course comes from the skill-gap
recommender against that person's actual gap. **The two halves of the project
become one decision.**

### B · Cross-population validation

Train on Population A. Restrict to features whose *meaning* genuinely transfers.
Score Population B. Grade against **Population B's own outcomes**.

| | Population A (held-out) | Population B (external) |
|---|--:|--:|
| ROC-AUC | 0.625 | **0.504** |
| PR-AUC | 0.275 | 0.104 |
| Base rate | 16.1% | 10.7% |

**The model does not transfer.** 0.504 is indistinguishable from a coin toss.

That is a negative result, and publishing it is worth more than a fudged positive:
it is independent confirmation of [section 3](#3--the-finding-that-changed-everything)
by a completely different method. The diagnosis is in the distribution shift:

| Feature | PSI | Pop A mean | Pop B mean |
|---|--:|--:|--:|
| YearsAtCompany | **5.42** | 7.0 years | 2.9 years |
| Age | 0.97 | 36.9 | 47.7 |
| WorkLifeBalance | 0.88 | 2.76 | 2.49 |

So **Population B receives no attrition score at all.** Those 3,000 rows carry
`attrition_probability: null` and a stated reason. A number indistinguishable from
a coin toss is worse than an honest blank, because someone will act on it.

<details>
<summary><b>What was excluded from the transfer contract, and why</b></summary>

A feature earns a place only if its *semantics* transfer, not merely its name:

* **`Department`, `JobRole`** — vocabularies are disjoint. Only "Sales" is shared
  across 3 vs 6 departments; 0 of 9 vs 31 job titles overlap. One-hot with
  `handle_unknown="ignore"` would silently zero them out and *look* like it worked.
* **`PerformanceRating`** — Population A only ever takes {3, 4}. There is no
  variation to learn from.
* **Likert rescaling** — B is 1–5, A is 1–4.
* **Support clipping** — 36.7% of Population B falls outside A's age range. Rows
  are clipped to the training support and the clipping is *counted and reported*,
  because extrapolating a fitted model beyond its support is exactly where silent
  nonsense begins.

</details>

### C · The budget-constrained action plan

A 0/1 knapsack across the workforce, solved greedily by ROI.

```
BUDGET 500,000 INR

  spend                             499,777   (100% utilised)
  employees at risk                     242
  employees funded                      235
  unfunded at risk                        7
  expected attritions prevented        32.5
  expected value retained           677,275
  return on investment                1.36×
```

> `GET /intelligence/action-plan?budget=500000`

### The guardrails — shipped in the product, not the README

* **Association, not causation.** The model learned which employees historically
  left, not what would have changed had a lever been pulled. Every response says so.
* **Feature independence is assumed.** True enough for single levers, weaker for
  combinations — which are flagged as more speculative.
* **Human in the loop.** SHAP is shown beside every recommendation, and a person
  approves before anything happens.
* **Protected attributes are never levers.** Enforced in code, asserted by a test.

An HR tool that overstates its certainty is a liability. Saying so is part of the
deliverable.

---

## 10 · API reference

`http://localhost:8000/docs` for interactive OpenAPI docs.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` · `/ready` | Liveness, and what is actually loaded |
| `POST` | `/api/v1/predict/attrition` | Score one employee, with SHAP reasons |
| `GET` | `/api/v1/predict/model` | Model version, metrics, threshold, git SHA |
| `GET` | `/api/v1/predict/log` | Prediction log — the drift feed |
| `GET` | `/api/v1/dashboard/summary` | Headline KPIs |
| `GET` | `/api/v1/dashboard/attrition-by-department` | Predicted **and actual** side by side |
| `GET` | `/api/v1/dashboard/engagement-by-department` | Engagement breakdown |
| `GET` | `/api/v1/dashboard/skill-gaps` | Organisation-wide gaps |
| `GET` | `/api/v1/dashboard/recommendations` | Per-employee upskilling |
| `GET` | `/api/v1/dashboard/model-quality` | Metrics + fairness + transfer validation |
| `GET` | `/api/v1/employees/{person_key}` | Full record — `A-101`, not `101` |
| `GET` | `/api/v1/employees/resolve/{id}` | Every person carrying a numeric ID |
| `GET` | `/api/v1/skills/crosswalk` | The 40 reviewed role mappings |
| `GET` | `/api/v1/skills/role-requirements` | What a role needs, both tiers |
| `GET` | `/api/v1/intelligence/counterfactual/{person_key}` | 🔑 What would reduce this person's risk |
| `GET` | `/api/v1/intelligence/action-plan?budget=X` | 🔑 The budget-constrained plan |
| `GET` | `/api/v1/intelligence/levers` | Available interventions + exclusions |

```bash
curl -X POST http://localhost:8000/api/v1/predict/attrition \
  -H 'Content-Type: application/json' \
  -d '{"Age":29,"Department":"Sales","JobRole":"Sales Executive","MonthlyIncome":4200,
       "OverTime":"Yes","JobSatisfaction":1,"WorkLifeBalance":1,"YearsAtCompany":2}'
```
```json
{ "attrition_probability": 0.788, "risk_band": "HIGH", "flagged": true,
  "model_version": "v1",
  "top_factors": [
    {"label": "OverTime",        "value": "Yes", "direction": "increases risk"},
    {"label": "WorkLifeBalance", "value": 1,     "direction": "increases risk"}],
  "caveat": "…never as an automatic decision." }
```

**Production details that matter:** the model loads **once** in the lifespan
handler (per-request loading would add ~800 ms to every call); Pydantic rejects
bad input with a 422 before it reaches the model; every response carries an
`X-Request-ID` for tracing.

---

## 11 · Dashboard

```
AI WORKFORCE INTELLIGENCE PLATFORM
──────────────────────────────────────────────────────────────────────
 Employees 4,470 │ High risk 58 │ Medium 184 │ Engagement 2.94/5 │ Gaps 4,470
──────────────────────────────────────────────────────────────────────
 Attrition risk │ Engagement │ Skill gaps │ Recommendations
 Retention Planner │ Employee 360 │ Model quality
```

Seven tabs. Three worth calling out:

* **Attrition risk** plots predicted *against actual* per department. If the bars
  diverge, the model is miscalibrated for that department and you should know
  before acting on it.
* **Retention Planner** — a budget slider that rebuilds the whole action plan live.
* **Employee 360** — drill into one person, see their SHAP reasons, their skill
  gaps, and what would actually reduce their risk.

The frontend calls the **API**, never the services directly. If the dashboard can
only see what the API exposes, the API cannot silently drift from being the real
contract.

---

## 12 · Testing

```bash
make test          # full suite + 80% coverage gate
make test-fast     # unit only, no coverage
```

**146 tests · 90% coverage.** Not just happy paths:

| Area | What is guarded |
|---|---|
| **The invariant** | The guard passes on a clean tree *and* catches a deliberate violation |
| **Validation** | Missing columns named individually; an engagement score of 80 fails |
| **Leakage** | The target and every label-adjacent column can never become a feature |
| **Determinism** | Derived skills are byte-identical across rebuilds |
| **Calibration** | The shipped model is smooth enough for counterfactuals to work |
| **Fairness** | Protected attributes and lever features are disjoint sets |
| **The ID collision** | `resolve/1001` returns *two different people*, live |
| **Honesty** | Population B never receives a fabricated risk score |
| **Budget** | The action plan never overspends, never proposes a risk-*raising* lever |
| **Reproducibility** | Every pipeline entry point runs and regenerates its artifacts |

All 17 notebooks are executed as part of verification — a notebook that does not
run is worse than no notebook.

---

## 13 · Monitoring & retraining

```bash
python -m hrai.monitoring.drift
```

Three questions, answered separately because they fail separately:

1. **Has the input moved?** PSI + Kolmogorov–Smirnov per watched feature. Needs no
   ground truth, so it is the earliest available warning.
2. **Have the predictions moved?** Cheaper still, and catches pipeline breakage
   that feature checks miss.
3. **Has performance dropped?** Only answerable once real outcomes arrive.

Evidently AI generates `docs/drift_report.html` alongside.

### The retraining rule, written down in advance

```
IF   drift PSI > 0.20
OR   F1 falls >15% below its own validated baseline
OR   6 months of new data
THEN retrain
```

**F1 is judged against the model's own baseline, not an absolute number.** This
model's F1 ceiling is ~0.40 by design — the threshold is deliberately low because
missing a leaver costs 24× an unnecessary conversation. An absolute floor of 0.70
would fire on every single run, and a team that sees an alarm every day stops
seeing it at all.

Full lifecycle: `new data → validation → training → evaluation → MLflow → human
approval → deploy`.

---

## 14 · What we would do next

Honest about what is not here:

* **The derived skill table is the weakest link.** It is defensible, deterministic
  and clearly labelled — but it is derived. A real skills inventory or an HRIS
  integration would replace it, and everything downstream would immediately get
  better.
* **The counterfactual engine assumes feature independence.** A real raise moves
  correlated features that this does not touch. Proper causal inference —
  double machine learning, or an actual A/B test on interventions — is the next
  step, and it is a large one.
* **Population B has no risk model.** The right fix is to train one *on* Population
  B using its own label, rather than transferring a model that measurably does not
  transfer.
* **Docker is written but unbuilt.** Docker was not installed on the development
  machine, so `Dockerfile` and `docker-compose.yml` are authored to spec but have
  not been executed. Verify with `docker compose up --build` before relying on them.
* **CI is not wired.** The pieces exist — `make lint`, `make test`,
  `make check-invariant` — but no GitHub Actions workflow runs them yet.

---

## Documentation index

| Document | What it holds |
|---|---|
| [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) | The full plan, all 8 findings, the risk register |
| [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) | **How to test everything** — copy-pasteable, with real expected outputs |
| [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) | Every requirement, and where it is satisfied |
| [`docs/adr/001-two-population-architecture.md`](docs/adr/001-two-population-architecture.md) | Why the two datasets are never joined |
| [`docs/adr/002-src-first-notebooks-as-narrative.md`](docs/adr/002-src-first-notebooks-as-narrative.md) | Why modules come before notebooks |
| [`docs/model_card.md`](docs/model_card.md) | Intended use, limitations, fairness, out-of-scope uses |
| `docs/data_dictionary.md` | Generated per-column profile of all five datasets |
| `docs/data_relationships.md` | Every table pair with its integrity result |
| `docs/transfer_validation.json` | The external validation result |
| `docs/fairness_audit.json` | Group metrics and interpretation |

---

<div align="center">

**Built following `HR_AI_Project_Build_Notes.docx` — its architecture, its 29 steps,
its toolchain.**

*Where the evidence contradicted the plan, the evidence won, and the reason is
written down.*

</div>
