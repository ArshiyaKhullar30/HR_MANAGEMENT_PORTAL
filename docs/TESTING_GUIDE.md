# How to Test This Application

A hands-on walkthrough. Every command is copy-pasteable, and every expected
output below is real output from this build — so if yours differs, something
genuinely changed.

**Time needed:** 2 minutes for the quick check, ~15 minutes for everything.

---

## Before you start

```bash
cd enterprise_hr_ai
conda activate hrai
```

Everything below assumes you are in the project root with the `hrai` environment
active.

---

## Level 1 · The 30-second check

Is anything actually working?

```bash
curl -s http://localhost:8000/health
```

**Expect:**
```json
{"status":"ok","model_loaded":true,"model_version":"v1","data_loaded":true}
```

| If you see | It means | Do this |
|---|---|---|
| `Connection refused` | The API is not running | `make api` in another terminal |
| `"model_loaded": false` | No trained model | `make train` |
| `"data_loaded": false` | Pipeline has not run | `make pipeline && make intelligence` |

If all three are `true`, the system is live. Skip to Level 3 if you just want to
click around.

---

## Level 2 · Run the automated tests

This is the real proof. 146 tests covering everything from schema contracts to
the live API.

```bash
make test
```

**Expect (takes ~90 seconds):**
```
146 passed in 90.63s
Required test coverage of 80% reached. Total coverage: 89.74%
```

Anything other than `146 passed` and a green coverage line means something broke.

<details>
<summary><b>Faster options while you are working</b></summary>

```bash
make test-fast                       # unit tests only, ~35s, no coverage gate
pytest tests/unit -q                 # same thing, directly
pytest tests/integration -q          # API + pipeline tests only
pytest -k "counterfactual" -q        # anything matching a keyword
pytest tests/unit/test_skills_and_gaps.py -v   # one file, verbose
```

</details>

### What the tests actually check

Worth knowing, so you can tell a real failure from a flaky one:

| Test area | What breaks it |
|---|---|
| `test_no_cross_population_join` | Someone joined the two employee datasets |
| `test_validation` | A cleaning rule stopped working |
| `test_features_and_model` | Leakage, or the model stopped being calibrated |
| `test_skills_and_gaps` | Derived skills stopped being deterministic |
| `test_counterfactual_and_fairness` | A protected attribute became an intervention lever |
| `test_api` | An endpoint changed its contract |
| `test_pipeline_end_to_end` | A pipeline step stopped being reproducible |

---

## Level 3 · Test the API by hand

Open **http://localhost:8000/docs** — every endpoint is there with a "Try it out"
button. Or use these:

### 3.1 · Score a high-risk employee

Young, overtime, unhappy, low tenure:

```bash
curl -s -X POST http://localhost:8000/api/v1/predict/attrition \
  -H 'Content-Type: application/json' \
  -d '{"Age":29,"Department":"Sales","JobRole":"Sales Executive",
       "MonthlyIncome":4200,"OverTime":"Yes","JobSatisfaction":1,
       "WorkLifeBalance":1,"YearsAtCompany":2}' | python -m json.tool
```

**Expect:** `attrition_probability` around **0.84**, `risk_band: "HIGH"`, and
`top_factors` naming OverTime and tenure.

### 3.2 · Score a low-risk employee

Senior, no overtime, satisfied, long tenure:

```bash
curl -s -X POST http://localhost:8000/api/v1/predict/attrition \
  -H 'Content-Type: application/json' \
  -d '{"Age":45,"Department":"Research & Development","JobRole":"Research Director",
       "MonthlyIncome":18000,"OverTime":"No","JobSatisfaction":4,"WorkLifeBalance":4,
       "YearsAtCompany":12,"TotalWorkingYears":20,"YearsInCurrentRole":8,
       "YearsWithCurrManager":8,"JobLevel":5,"StockOptionLevel":2}' | python -m json.tool
```

**Expect:** around **0.008**, `risk_band: "LOW"`, `flagged: false`.

> **The point of running both:** 84% vs 0.8% on the same model. If both profiles
> return similar numbers, the model is not discriminating and something is wrong.

### 3.3 · Confirm bad input is rejected

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/api/v1/predict/attrition \
  -H 'Content-Type: application/json' \
  -d '{"Age":250,"Department":"Sales","JobRole":"Sales Executive","MonthlyIncome":4200,
       "OverTime":"Yes","JobSatisfaction":1,"WorkLifeBalance":1,"YearsAtCompany":2}'
```

**Expect:** `422`. A 200 here would mean invalid data reached the model.

Other things that should also return `422`:

```bash
# OverTime must be Yes/No
-d '{...,"OverTime":"Sometimes"}'
# JobSatisfaction is a 1-4 scale
-d '{...,"JobSatisfaction":9}'
# You cannot have been in your role longer than at the company
-d '{...,"YearsAtCompany":2,"YearsInCurrentRole":99}'
```

### 3.4 · The ID collision — the most important single check

```bash
curl -s http://localhost:8000/api/v1/employees/resolve/1001 | python -m json.tool
```

**Expect two different people:**
```
A-1001  Research & Development   Laboratory Technician
B-1001  Software Engineering     Software Engineer
```

> Employee ID `1001` exists in both source datasets — for two unrelated people at
> two different companies. If this ever returns **one** match, the two populations
> have been merged and every number in the system is suspect.

### 3.5 · Confirm Population B gets no fabricated score

```bash
curl -s http://localhost:8000/api/v1/employees/B-1001 | python -m json.tool
```

**Expect:** `"attrition_probability": null` and `"risk_band": "UNAVAILABLE"`,
with a `risk_unavailable_reason` explaining that the model does not transfer
(externally validated at ROC-AUC 0.50 — chance).

A number here would be a bug, not a feature.

### 3.6 · The Retention ROI Copilot

**What would reduce one person's risk:**

```bash
curl -s http://localhost:8000/api/v1/intelligence/counterfactual/A-622 | python -m json.tool
```

**Expect** baseline risk ~93.5% and a ranked list, best return first:

| Intervention | Risk drop | Cost | ROI |
|---|--:|--:|--:|
| Remove mandatory overtime | 12.2% | 2,808 | 0.61 |
| Work-life balance programme | 1.6% | 1,170 | 0.19 |
| Targeted upskilling | 0.7% | 576 | 0.16 |

**The whole-workforce plan:**

```bash
curl -s "http://localhost:8000/api/v1/intelligence/action-plan?budget=500000" \
  | python -c "import json,sys; d=json.load(sys.stdin); [print(f'{k:32} {d[k]}') for k in
    ['employees_at_risk','employees_covered','spend','expected_attritions_prevented',
     'expected_value_retained','return_on_investment']]"
```

**Expect:**
```
employees_at_risk                242
employees_covered                235
spend                            499777.1
expected_attritions_prevented    32.53
expected_value_retained          677275.02
return_on_investment             1.36
```

Try changing `budget=` — `100000`, `2000000` — and watch coverage and ROI move.
`spend` must never exceed `budget`.

### 3.7 · Sweep every endpoint at once

```bash
for p in /health /ready \
  /api/v1/dashboard/summary /api/v1/dashboard/attrition-by-department \
  /api/v1/dashboard/engagement-by-department /api/v1/dashboard/skill-gaps \
  /api/v1/dashboard/recommendations /api/v1/dashboard/departments \
  /api/v1/dashboard/model-quality /api/v1/predict/model /api/v1/predict/log \
  /api/v1/employees/A-622 /api/v1/employees/resolve/1001 \
  /api/v1/skills/crosswalk /api/v1/skills/role-requirements \
  /api/v1/intelligence/levers /api/v1/intelligence/counterfactual/A-622 \
  "/api/v1/intelligence/action-plan?budget=300000"; do
  printf "%s  %s\n" "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8000$p")" "$p"
done
```

**Expect:** `200` on every line.

---

## Level 4 · Test the dashboard

Open **http://localhost:8501**.

Not running? Start it (in its own terminal — it keeps running):

```bash
make dashboard
```

Stop it with `Ctrl+C`, or from anywhere: `pkill -f "streamlit run"`.

### What to check on each tab

| Tab | Do this | Expect |
|---|---|---|
| **Top KPI row** | Just look | 4,470 employees · 58 high risk · 184 medium · engagement 2.94/5 |
| **Attrition risk** | Compare the two bars per department | Predicted and actual nearly equal — Sales 0.211 vs 0.206. Divergence means miscalibration |
| **Engagement** | Look at the ordering | Production lowest (~2.91), Executive Office highest (~3.38) |
| **Skill gaps** | Look at the top rows | SAP, MATLAB, SharePoint — and an amber "skills are derived" banner |
| **Recommendations** | Scan the table | Every row has a course; costs come from real training records |
| **Retention Planner** | Set a budget, click **Build action plan** | Funded/unfunded counts, ROI, and a ranked intervention table |
| **Employee 360** | Type `A-622`, press Enter | Risk, SHAP reasons, skill gaps, and what would reduce their risk |
| **Employee 360** | Now type `B-1001` | Risk shows **Unavailable** with a warning explaining why |
| **Model quality** | Scroll through | Metrics, drivers, fairness (2 flagged with explanations), transfer validation |

### Three things worth deliberately confirming

1. **Sidebar department filter** — change it and watch the KPI row update.
2. **Budget slider** — move it and rebuild. `spend` should never exceed budget.
3. **The caveats are visible.** Every counterfactual panel ends with a red box
   saying this is association, not causation. That is intentional and must stay.

---

## Level 5 · Test reproducibility

The strongest test: delete everything and prove it rebuilds identically.

```bash
# 1 — record the current checksums
python -c "
from hrai.utils.io import read_manifest; import json
json.dump({k:v['content_sha256'] for k,v in read_manifest().items()}, open('/tmp/before.json','w'))
print('saved', len(read_manifest()), 'checksums')"

# 2 — delete every generated artifact
find data/processed data/interim models -mindepth 1 -delete
rm -f mlflow.db

# 3 — rebuild from the raw CSVs (~4 minutes)
make all

# 4 — compare
python -c "
from hrai.utils.io import read_manifest; import json
before = json.load(open('/tmp/before.json'))
after = {k: v['content_sha256'] for k, v in read_manifest().items()}
same = [k for k in before if before.get(k) == after.get(k)]
print(f'byte-identical: {len(same)}/{len(before)}')
for k in before:
    if k in after and before[k] != after[k]: print('  differs:', k)"
```

**Expect:** `byte-identical: 7/8`, with only `employee_intelligence` differing —
because it embeds the model version string, which increments each time you train.
Every actual value, including all 1,470 attrition probabilities, stays identical.

> **Careful with step 2 in zsh:** `rm -rf data/processed/*` silently aborts the
> whole command if any glob matches nothing. Use the `find ... -delete` form above.

---

## Level 6 · Test the individual pipeline steps

Each step runs on its own and writes a report you can inspect.

```bash
make profile         # Step 01 → docs/data_dictionary.md, docs/findings.json
make validate        # Step 02 → docs/validation_report.md
make clean-data      # Step 03 → data/processed/ + strict re-validation
make relationships   # Step 04 → docs/data_relationships.md
make train           # Steps 05-09 → models/vN/ + SHAP report
make crosswalk       # Step 11 → the role → O*NET mapping
make intelligence    # Steps 10-16 → the unified employee table
make transfer        # external validation → docs/transfer_validation.json
make fairness        # bias audit → docs/fairness_audit.json
make monitor         # drift + retraining decision → docs/monitoring_report.json
```

### Spot-checks worth doing

**Validation should fail on raw data — that is correct:**
```bash
make validate && head -20 docs/validation_report.md
```
Expect `hr_performance_engagement` **FAIL** with **1,198** violations. Those are
the real defects; cleaning is what fixes them.

**Check the invariant guard is active:**
```bash
make check-invariant
```
Expect `OK — no cross-population join.`

**Prove the guard actually catches something:**
```bash
cat > src/hrai/_test_violation.py <<'EOF'
import pandas as pd
att = pd.read_csv("data/raw/employee_attrition.csv")
eng = pd.read_csv("data/raw/hr_performance_engagement.csv")
bad = att.merge(eng, left_on="EmployeeNumber", right_on="Employee ID")
EOF
make check-invariant          # should FAIL loudly
rm src/hrai/_test_violation.py
make check-invariant          # should pass again
```

---

## Level 7 · Test the notebooks

```bash
make notebooks
```

Runs all 17 top-to-bottom. Takes ~10 minutes. Any error means a notebook has
drifted from the code it imports.

Single notebook:
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/07_model_comparison.ipynb
```

Or just open them: `jupyter lab notebooks/`

---

## Level 8 · Code quality

```bash
make lint              # ruff + black, no changes made
make format            # auto-fix
```

**Expect:** `All checks passed!` from ruff, silence from black.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` on :8000 | API not running | `make api` |
| Dashboard shows "Cannot reach the API" | API down, or wrong URL | Start the API; or set `HRAI_API_URL` |
| `ModuleNotFoundError: hrai` | Package not installed | `pip install -e . --no-deps` |
| `XGBoostError: libomp.dylib` | Missing OpenMP (macOS) | `conda install -c conda-forge llvm-openmp` |
| `FileNotFoundError: No trained model` | Never trained | `make train` |
| `503` from dashboard endpoints | Intelligence table missing | `make intelligence` |
| `make` looks frozen | It is not — long steps just take time | Watch for the JSON log lines |
| Port already in use | An old process is still up | `pkill -f "uvicorn app.main"` / `pkill -f "streamlit run"` |

### Starting and stopping cleanly

```bash
# start (each in its own terminal)
make api
make dashboard

# stop
pkill -f "uvicorn app.main"
pkill -f "streamlit run"

# what is running?
lsof -i :8000 -i :8501
```

---

## The five-minute version

If you only do one thing:

```bash
make test                                          # 146 passed, 90% coverage
curl -s http://localhost:8000/health               # all three flags true
curl -s http://localhost:8000/api/v1/employees/resolve/1001   # TWO people
open http://localhost:8501                         # click every tab
```

Those four cover: the code works, the service is live, the central data
invariant holds, and the UI renders.
