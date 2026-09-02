# Model Card — Attrition Prediction Model

| | |
|---|---|
| **Model** | Attrition Prediction Model, version `v1` |
| **Algorithm** | Logistic Regression + sigmoid (Platt) calibration |
| **Owner** | Enterprise HR AI — Workforce Intelligence Platform |
| **Trained on** | `employee_attrition.csv` (Population A) — 1,470 employees |
| **Framework** | scikit-learn 1.9.0, Python 3.11 |
| **Artifact** | `models/v1/attrition_pipeline.joblib` |

---

## 1. Intended use

**What it is for.** Ranking employees by estimated attrition risk so an HR team
can decide who to talk to first, and — via the Retention ROI Copilot — which
intervention to consider for them.

**Who should use it.** HR business partners and people-analytics teams, with a
human reviewing every output before any action.

**What it is explicitly NOT for.**

- Any automated or semi-automated employment decision — termination, promotion,
  compensation, performance rating, or hiring.
- Individual performance assessment. This model estimates *departure risk*, not
  quality of work.
- Scoring anyone outside Population A. See §5.
- Presentation to the employee as a statement about their intentions. It is a
  statistical estimate from historical patterns.

---

## 2. How it was built

| Decision | Choice | Why |
|---|---|---|
| Candidates | Logistic Regression, Random Forest, XGBoost | The three named in the project brief |
| Class balance | `class_weight="balanced"` / `scale_pos_weight` | XGBoost needs its own mechanism, or the comparison is not fair |
| Validation | Repeated stratified 5-fold × 3 | 237 positives makes a single split too high-variance to compare on |
| Selection metric | PR-AUC, recall as tie-break | PR-AUC is the honest metric under imbalance; a missed leaver is the expensive error |
| Calibration | Sigmoid, selected on evidence | See §3 |
| Threshold | 0.08, by expected cost | See §4 |
| Preprocessing | Entirely inside the pipeline object | Makes train/serve skew structurally impossible |

**Model selection result.** Logistic Regression won on PR-AUC (0.637) over
XGBoost (0.563) and Random Forest (0.549). The project brief expected XGBoost to
win. It did not: 1,470 rows with a largely linear signal is the regime where a
regularised linear model beats boosting. The measured result is reported rather
than the expected one.

---

## 3. Calibration

Both calibration methods were evaluated on out-of-fold predictions:

| Method | OOF Brier | Saturated fraction | Distinct values |
|---|--:|--:|--:|
| **sigmoid** (selected) | 0.0961 | 0.0% | 1,172 |
| isotonic | 0.0930 | 3.6% | 614 |

Isotonic scored marginally better on Brier and was rejected anyway. It is a step
function, so a band of raw scores maps to exactly 1.0 — and a saturated
probability cannot respond to a counterfactual perturbation at all, which would
silently break the intervention engine. Smoothness was the deciding criterion,
and the comparison is stored in the model metadata.

---

## 4. Operating threshold

Chosen by expected cost, not by maximising F1:

- **False negative** = losing an employee = 6 × monthly salary (replacement cost)
- **False positive** = one unnecessary retention conversation = 0.25 × monthly salary

Both scale with the individual's salary, so the ratio is constant at **24:1**.

| | |
|---|--:|
| Empirical optimum (grid search) | **0.08** |
| Analytic Bayes optimum | 0.04 |
| Expected cost saving vs 0.5 | ₹2,612,592 |

The two agreeing is a useful check that the probabilities are genuinely
calibrated rather than merely well-ordered.

**A second, capacity-bounded threshold** exists for the dashboard watchlist. Cost
optimality correctly says "flag many people"; that is right economics and a
useless work queue. The two questions get two answers.

---

## 5. Performance

### Held-out test set (294 employees, Population A)

| Metric | Value |
|---|--:|
| ROC-AUC | 0.809 |
| PR-AUC | 0.534 |
| Precision | 0.261 |
| Recall | **0.894** |
| F1 | 0.404 |
| Brier | 0.104 |

**Accuracy is not reported.** At a 16.1% positive rate, predicting "stays" for
everyone scores 83.9% while being useless. Including accuracy would invite
exactly the wrong comparison.

Low precision is deliberate: the 24:1 cost asymmetry means over-flagging is much
cheaper than missing someone.

### Calibration by department

| Department | Mean predicted | Actual rate |
|---|--:|--:|
| Sales | 0.211 | 0.206 |
| Human Resources | 0.181 | 0.191 |
| Research & Development | 0.136 | 0.138 |

### External validation — the model does NOT generalise

Trained on Population A, restricted to transferable features, and scored against
Population B's own `Voluntarily Terminated` outcomes:

| | Population A (held-out) | Population B (external) |
|---|--:|--:|
| ROC-AUC | 0.625 | **0.504** |
| PR-AUC | 0.275 | 0.104 |

**0.504 is chance.** The cause is severe distribution shift — tenure PSI 5.42
(Population A averages 7.0 years, B averages 2.9). These are different workforces.

**Consequence:** the platform withholds a risk score for Population B entirely.
Those 3,000 records return `attrition_probability: null` with a stated reason.

---

## 6. Fairness

Group metrics at the operating threshold, minimum group size 30, tolerance 0.10:

| Attribute | Equal-opportunity diff | Predictive-equality diff | Max calibration gap | Verdict |
|---|--:|--:|--:|---|
| Gender | 0.032 | 0.014 | 0.004 | Within tolerance |
| MaritalStatus | 0.101 | 0.233 | 0.022 | **Flagged** |
| Age band | 0.206 | 0.314 | 0.040 | **Flagged** |

**Interpretation.** Every group's calibration gap is under 0.04, so the model is
well-calibrated *within* each group. The differing flag rates follow from
genuinely different base rates (18–29 leave at 27.9%; 50+ at 13.3%). Calibration
and equalised odds cannot both hold when base rates differ — that is a theorem,
not a defect. What the model is doing is reporting a real difference, not adding
one.

**That is not a licence to ignore it.** Flagged groups must be routed to human
review, and the score must never drive an automatic decision.

**Protected attributes can never be intervention levers.** `Age`, `Gender` and
`MaritalStatus` are excluded from the counterfactual lever set structurally, and
`tests/unit/test_counterfactual_and_fairness.py` asserts the two sets are
disjoint.

---

## 7. Limitations

1. **Association, not causation.** The model learned which employees historically
   left. It did not establish that changing a feature changes the outcome. Every
   counterfactual output states this.
2. **Feature independence is assumed** in counterfactuals. A real raise moves
   correlated features this does not touch. Single-lever estimates are the most
   trustworthy; combinations are flagged as more speculative.
3. **Small positive class.** 237 positive cases. Metrics carry real variance —
   hence repeated CV, and hence reporting mean ± std rather than a point estimate.
4. **Single-company training data.** Externally validated as non-transferable
   (§5). Do not deploy against a different workforce without revalidating.
5. **Historical labels may encode historical bias.** Fairness parity here means
   the model adds no measurable disparity, not that the underlying process was fair.
6. **Skill data is derived, not observed.** No source dataset records individual
   skills. Profiles are deterministically derived from role requirements, tenure,
   performance and training history, and flagged `is_derived` everywhere.

---

## 8. Reproducibility

| | |
|---|---|
| Seed | 42, threaded through every split, model and generator |
| Data checksums | Recorded in `models/v1/metadata.json` |
| Git SHA | Recorded at training time |
| Library versions | Recorded at training time |
| Rebuild | `make pipeline && make train` |
| Experiment tracking | MLflow (`sqlite:///mlflow.db`) |

---

## 9. Monitoring and retraining

Retrain when **any** of:

- Input drift PSI > 0.20 on a watched feature
- F1 falls more than 15% below its own validated baseline (0.404 → limit 0.343)
- Six months of new data accumulate

F1 is judged relative to the model's own baseline rather than an absolute floor:
this model's F1 ceiling is set by a deliberately low operating threshold, not by
how well it separates classes. An absolute floor of 0.70 would fire every run.

Lifecycle: `new data → validation → training → evaluation → MLflow → human
approval → deploy`.

---

## 10. Contact and escalation

Raise an issue in this repository if the model appears to be behaving
unexpectedly. **Stop using the score for decisions** if:

- the fairness audit shows a calibration gap above 0.10 for any group, or
- drift monitoring reports severe shift on more than one watched feature, or
- observed F1 falls below 0.343.
