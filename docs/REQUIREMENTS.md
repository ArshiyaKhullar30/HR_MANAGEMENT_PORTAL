# Requirements & Traceability

Every requirement in `HR_AI_Project_Build_Notes.docx`, and exactly where it is
satisfied in this repository. Deviations are listed in §6, each with its evidence.

Status key: **✅ Done** · **⚠️ Done, with a documented deviation** · **📋 Written, not executed**

---

## 1 · Functional requirements

The brief's short version: *"an HR system that looks at employee data and tells
the company three things — who is at risk of quitting, where the skill gaps are
across the organisation, and what each employee should learn next."*

| # | Requirement | Status | Where |
|---|---|:--:|---|
| F1 | Predict who is at risk of leaving | ✅ | `src/hrai/ml/train.py`, `POST /predict/attrition` |
| F2 | Explain *why* an employee is flagged | ✅ | `src/hrai/ml/explain.py` — SHAP global + local |
| F3 | Bucket employees into risk levels | ✅ | `hrai.ml.evaluate.risk_band` — HIGH/MEDIUM/LOW |
| F4 | Track engagement across the organisation | ✅ | `src/hrai/intelligence/engagement.py` |
| F5 | Identify skill gaps per employee | ✅ | `src/hrai/skills/gap.py` |
| F6 | Identify organisation-wide skill gaps | ✅ | `hrai.skills.gap.organisation_skill_gaps` |
| F7 | Recommend what each employee should learn next | ✅ | `src/hrai/skills/recommend.py` |
| F8 | One unified record per employee | ✅ | `src/hrai/intelligence/employee_table.py` |
| F9 | Web app so results are viewable outside a notebook | ✅ | `frontend/Home.py` + `app/main.py` |

---

## 2 · The 29-step checklist

### Day 1 — Data foundation

| # | Step | Status | Implementation | Notebook |
|:--:|---|:--:|---|---|
| 01 | Data Understanding | ✅ | `hrai/profiling/profiler.py`, `run.py` | `01_data_understanding` |
| 02 | Data Validation | ⚠️ | `hrai/validation/schemas.py`, `runner.py` | `02_data_validation` |
| 03 | Data Cleaning | ✅ | `hrai/cleaning/{attrition,engagement,onet,text}.py` | `03_data_cleaning` |
| 04 | Data Relationships | ⚠️ | `hrai/profiling/relationships.py` | `04_data_relationships` |

### Day 2 — Machine learning

| # | Step | Status | Implementation | Notebook |
|:--:|---|:--:|---|---|
| 05 | Feature Engineering | ✅ | `hrai/features/{engineering,pipeline}.py` | `05_feature_engineering` |
| 06 | Baseline Model | ⚠️ | `hrai/ml/train.py` | `06_baseline_model` |
| 07 | Model Comparison | ⚠️ | `hrai.ml.train.compare_models` | `07_model_comparison` |
| 08 | SHAP Explainability | ✅ | `hrai/ml/explain.py` | `08_model_explainability` |
| 09 | Model Versioning + MLflow | ✅ | `hrai/ml/{registry,tracking}.py` | `09_model_versioning` |

### Day 3 — Workforce intelligence

| # | Step | Status | Implementation | Notebook |
|:--:|---|:--:|---|---|
| 10 | Engagement Intelligence | ⚠️ | `hrai/intelligence/engagement.py` | `10_engagement_intelligence` |
| 11 | Role Intelligence | ⚠️ | `hrai/skills/{crosswalk,ontology}.py` | `11_role_intelligence` |
| 12 | Employee Skills | ⚠️ | `hrai/skills/employee_skills.py` | `12_employee_skills` |
| 13 | Skill Gap Engine | ⚠️ | `hrai/skills/gap.py` | `13_skill_gap_engine` |
| 14 | Organization Skill Gap | ⚠️ | `hrai.skills.gap.organisation_skill_gaps` | `14_organization_skill_gap` |
| 15 | Recommendation Engine | ✅ | `hrai/skills/recommend.py` | `15_recommendation_engine` |
| 16 | Employee Intelligence Layer | ⚠️ | `hrai/intelligence/employee_table.py` | `16_employee_intelligence` |

### Day 4 — Application

| # | Step | Status | Implementation |
|:--:|---|:--:|---|
| 17 | Refactor into modules | ⚠️ | `src/` + `app/` — done from day one; see ADR 002 |
| 18 | FastAPI Backend | ✅ | `app/main.py`, `app/api/*.py` — all 6 endpoints plus extras |
| 19 | API Input Validation | ✅ | `app/validation/{employee,engagement}_schema.py` |
| 20 | Logging | ✅ | `hrai/utils/logger.py` — structured JSON with run IDs |
| 21 | Prediction Logging | ✅ | `app/services/prediction_log.py` → `data/predictions/` |
| 22 | Unit Testing | ✅ | `tests/` — 146 tests, 90% coverage |
| 23 | Streamlit Dashboard | ✅ | `frontend/Home.py` — 7 tabs |

### Later — Enterprise hardening

| # | Step | Status | Implementation |
|:--:|---|:--:|---|
| 24 | Docker | 📋 | `Dockerfile`, `docker-compose.yml`, `.dockerignore` — written, **not built** (Docker absent from the dev machine) |
| 25 | Data Drift Monitoring | ✅ | `hrai/monitoring/drift.py` — PSI + KS, plus Evidently |
| 26 | Model Performance Monitoring | ✅ | `hrai.monitoring.drift.performance_monitor` |
| 27 | Retraining Strategy | ⚠️ | `hrai.monitoring.drift.retraining_decision` |
| 28 | Documentation | ✅ | `README.md`, `docs/` — model card, ADRs, generated reports |
| 29 | Deployment | 📋 | Architecture and compose stack written; CI not wired |

---

## 3 · Named API endpoints

All six from the brief, plus additions for the Copilot layer.

| Brief specifies | Status | Actual route |
|---|:--:|---|
| `POST /predict/attrition` | ✅ | `POST /api/v1/predict/attrition` |
| `GET /dashboard/summary` | ✅ | `GET /api/v1/dashboard/summary` |
| `GET /dashboard/attrition-by-department` | ✅ | `GET /api/v1/dashboard/attrition-by-department` |
| `GET /dashboard/skill-gaps` | ✅ | `GET /api/v1/dashboard/skill-gaps` |
| `GET /dashboard/recommendations` | ✅ | `GET /api/v1/dashboard/recommendations` |
| `GET /employees/{employee_id}` | ⚠️ | `GET /api/v1/employees/{person_key}` — see D8 |

Additional: `/health`, `/ready`, `/predict/model`, `/predict/log`,
`/dashboard/engagement-by-department`, `/dashboard/model-quality`,
`/dashboard/departments`, `/employees/resolve/{id}`, `/skills/crosswalk`,
`/skills/role-requirements`, `/intelligence/counterfactual/{person_key}`,
`/intelligence/action-plan`, `/intelligence/levers`.

---

## 4 · Named test cases

| Brief specifies | Status | Test |
|---|:--:|---|
| Missing required column is caught | ✅ | `test_missing_required_column_is_caught` |
| Invalid engagement score is rejected | ✅ | `test_invalid_engagement_score_is_rejected` |
| Prediction returns a real probability | ✅ | `test_prediction_returns_a_real_probability` |
| Risk level assigned correctly from probability | ✅ | `test_risk_band_boundaries` |
| Skill gap calculation matches expected output | ✅ | `test_technical_gap_matches_the_expected_subtraction` |
| API returns expected status codes | ✅ | `test_read_endpoints_return_200` + 6 parametrised rejection cases |

---

## 5 · Toolchain

Every library named in the brief, used as named.

| Layer | Required | Version |
|---|---|---|
| Core | pandas, numpy, matplotlib | 2.3.3, 2.4.6, 3.11.1 |
| Storage | pyarrow (Parquet) | 25.0.1 |
| Validation | pandera, pydantic | 0.24.0, 2.13.4 |
| ML | scikit-learn, xgboost, shap, joblib | 1.9.0, 3.2.0, 0.51.0, 1.5.3 |
| Tracking | mlflow | 3.15.2 |
| NLP | sentence-transformers (`all-MiniLM-L6-v2`) | 5.7.0 |
| API | fastapi, uvicorn | 0.141.1, 0.52.4 |
| Frontend | streamlit, plotly | 1.62.0, 5.24.1 |
| Monitoring | evidently | 0.7.21 |
| Testing | pytest, pytest-cov, httpx | 8.4.2, 7.1.0, 0.28.1 |
| Quality | black, ruff, pre-commit, nbstripout | 25.12.0, 0.16.5, 4.6.2, 0.9.1 |
| Packaging | Docker, docker-compose | written, not built |

Fully resolved environment: `requirements.lock.txt` (243 packages).

---

## 6 · Deviations from the brief

Every deviation, its evidence, and why it was made.

| # | Brief says | We did | Evidence |
|:--:|---|---|---|
| **D1** | The two employee datasets join 1:1 on EmployeeID (Step 04) | **Never joined.** Two-population architecture | Gender agrees 48.6%, age 6.0% on the 753 shared IDs. ADR 001 |
| **D2** | Plain-pandas asserts for MVP, Pandera later (Step 02) | **Pandera from the start** | The brief names Pandera as the destination; throwaway asserts cost more than they save |
| **D3** | Engagement range check 0–100 (Step 02) | **1–5 Likert** | Observed range is [1,5]. A 0–100 rule passes every row |
| **D4** | Skill gaps from `essential_skills.csv` | **Two-tier ontology** | That file holds 10 cognitive skills, not tools. Technical skills come from `software_skills.csv` |
| **D5** | Employee skills exist or are invented | **Deterministically derived, flagged** | No dataset records them. The brief pre-authorised the fallback |
| **D6** | Roles map to O\*NET | **Semantic crosswalk + full human review** | 0 of 40 exact matches. Machine mapped CIO → *Editors* at 0.21 |
| **D7** | Org gap severity by count (100+/50+) | **Percentage of workforce** | Same rule, scale-invariant |
| **D8** | `GET /employees/{employee_id}` | **`{person_key}`**, plus `/resolve/{id}` | A numeric ID is ambiguous across populations. Both people are returned |
| **D9** | Notebooks first, refactor on Day 4 (Step 17) | **src-first** | The refactor step is where projects collapse. ADR 002 |
| **D10** | Single train/test split (Step 06) | **Repeated stratified CV** | 237 positives makes one split too noisy to compare on |
| **D11** | XGBoost expected to win (Step 07) | **Logistic Regression won** | PR-AUC 0.637 vs 0.563, measured on identical folds |
| **D12** | Threshold implicit | **Cost-optimal, plus a capacity threshold** | The brief asked for "the actual cost of mistakes"; made quantitative |
| **D13** | Retrain if F1 below a threshold (Step 27) | **Relative to the model's own baseline** | This model's F1 ceiling is ~0.40 by design; an absolute 0.70 floor would fire every run |

---

## 7 · Added beyond the brief

| Addition | Why |
|---|---|
| **Retention ROI Copilot** — counterfactuals, priced levers, budget knapsack | Answers the question an HR director actually has |
| **Cross-population external validation** | Population B's own label makes real external validation possible |
| **Fairness audit** | An HR model influencing decisions about people needs a bias check |
| **The invariant guard** (pre-commit + CI + test) | Makes the D1 finding impossible to undo by accident |
| **`person_key`** | Structural fix for the 753-ID collision |
| **Determinism contract** | One seed, hash-based variation, asserted by tests |
| **Checksummed artifacts** | Traceability from any result back to exact input bytes |
| **Two operating thresholds** | Cost optimality and operational capacity are different questions |
| **Evidence-based calibration selection** | Isotonic saturates and would break counterfactuals |

---

## 8 · Non-functional requirements

| Requirement | How it is met |
|---|---|
| **Reproducibility** | One seed; hash-based variation; checksummed artifacts; pinned + locked deps; a test asserts byte-identical rebuilds |
| **Scalability** | Parquet columnar storage; model loaded once at startup; cached read model; date-partitioned prediction log |
| **Maintainability** | src-first; config-driven (zero hard-coded paths); one definition per validation rule; 90% coverage |
| **Observability** | Structured JSON logs with run IDs; request correlation; prediction log; drift monitoring |
| **Security & privacy** | PII dropped in cleaning; `nbstripout` mandatory in pre-commit; `data/` gitignored; non-root Docker user |
| **Correctness** | Pandera contracts at both stages; leakage register enforced in code; the invariant guard |
| **Honesty** | Risk withheld where the model does not transfer; derived data labelled everywhere; caveats in the UI, not just the docs |
