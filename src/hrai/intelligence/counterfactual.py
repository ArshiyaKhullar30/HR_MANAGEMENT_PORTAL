"""Counterfactual retention planning — the Retention ROI Copilot.

Every capable version of this project answers "who is likely to leave, and
why?". That is where the Build Notes stop, and it is where the field stops. It
is not the question an HR director actually has. Theirs is: *"I have a retention
budget. Who do I spend it on, on what, and what do I get back?"*

This module answers that, in four steps:

1. **Perturb.** For one employee, change a single *actionable* lever
   (OverTime -> No, WorkLifeBalance +1, promotion, a raise, ...) and push the
   modified row back through the fitted pipeline. The change in predicted risk
   is that lever's effect.
2. **Price.** Each lever has a cost, expressed as a multiple of that employee's
   own monthly salary so it scales with seniority. Training is priced from the
   *real* `Training Cost` distribution in the engagement data.
3. **Value.** Expected value saved = reduction in risk x replacement cost.
4. **Allocate.** Across the whole workforce, choose the set of interventions
   that maximises expected value retained within a fixed budget — a knapsack,
   solved greedily by ROI.

Three guardrails, none optional:

* **Protected attributes are never levers.** Age, gender and marital status are
  excluded structurally, and a test asserts it. The system cannot propose an
  intervention on who someone is.
* **This is association, not causation.** The model learned that people who
  work overtime leave more often; it did not establish that stopping the
  overtime stops the leaving. Every response says so.
* **Feature independence is assumed.** Changing MonthlyIncome does not update
  the correlated features a real raise would move. Single-lever estimates are
  the most trustworthy; combinations are flagged as more speculative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import pandas as pd

from hrai.ml.registry import load_model
from hrai.utils.config import get
from hrai.utils.logger import get_logger

log = get_logger(__name__)

CAVEAT = (
    "Association-based decision support, not causal inference. The model learned "
    "which employees historically left, not what would have changed if a lever had "
    "been pulled. Counterfactuals also assume features move independently. "
    "Treat as a ranked prompt for a human conversation, never as an automatic decision."
)


@dataclass
class Lever:
    name: str
    label: str
    feature: str
    rationale: str
    set_value: Any = None
    delta: float | None = None
    max_value: float | None = None
    pct: float | None = None
    cost_multiple_monthly: float | None = None
    cost_absolute: float | None = None

    def applies_to(self, row: pd.Series) -> bool:
        """False when the lever is irrelevant or already exhausted for this person."""
        if self.feature not in row.index:
            return False
        current = row[self.feature]
        if pd.isna(current):
            return False
        if self.set_value is not None:
            return current != self.set_value
        if self.delta is not None and self.max_value is not None:
            return float(current) < float(self.max_value)
        return True

    def apply(self, row: pd.Series) -> pd.Series:
        out = row.copy()
        current = out[self.feature]
        if self.set_value is not None:
            out[self.feature] = self.set_value
        elif self.pct is not None:
            out[self.feature] = type(current)(float(current) * (1.0 + self.pct))
        elif self.delta is not None:
            new_value = float(current) + self.delta
            if self.max_value is not None:
                new_value = min(new_value, float(self.max_value))
            out[self.feature] = type(current)(new_value)
        return out

    def cost(self, monthly_income: float, training_cost: float) -> float:
        if self.name == "targeted_training":
            return round(float(self.cost_absolute or training_cost), 2)
        if self.cost_absolute is not None:
            return round(float(self.cost_absolute), 2)
        return round(float(monthly_income) * float(self.cost_multiple_monthly or 0.0), 2)


def load_levers() -> list[Lever]:
    """Levers from config, with protected attributes structurally excluded."""
    protected = set(get("retention_roi.protected_attributes", []))
    levers: list[Lever] = []
    for spec in get("retention_roi.levers", []) or []:
        if spec["feature"] in protected:
            # Defence in depth: config is already correct, and a test asserts it,
            # but a lever on a protected attribute must never be constructible.
            log.error(
                "refusing lever on a protected attribute",
                extra={"lever": spec.get("name"), "feature": spec["feature"]},
            )
            continue
        levers.append(
            Lever(
                name=spec["name"],
                label=spec.get("label", spec["name"]),
                feature=spec["feature"],
                rationale=spec.get("rationale", ""),
                set_value=spec.get("set"),
                delta=spec.get("delta"),
                max_value=spec.get("max"),
                pct=spec.get("pct"),
                cost_multiple_monthly=spec.get("cost_multiple_monthly"),
                cost_absolute=spec.get("cost_absolute"),
            )
        )
    return levers


def median_training_cost() -> float:
    """Real observed cost of a training intervention, from the engagement data."""
    try:
        from hrai.skills.recommend import build_course_catalogue

        return float(build_course_catalogue()["median_cost"].median())
    except Exception:  # noqa: BLE001 - never let pricing break a prediction
        return 576.0


@dataclass
class Intervention:
    lever: str
    label: str
    feature: str
    from_value: Any
    to_value: Any
    baseline_risk: float
    new_risk: float
    risk_reduction: float
    cost: float
    expected_value_saved: float
    roi: float
    recommended_course: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class EmployeePlan:
    employee_id: int
    person_key: str
    baseline_risk: float
    risk_band: str
    replacement_cost: float
    single_lever: list[Intervention] = field(default_factory=list)
    best_combination: dict[str, Any] | None = None
    caveat: str = CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "person_key": self.person_key,
            "baseline_risk": round(self.baseline_risk, 4),
            "risk_band": self.risk_band,
            "replacement_cost": round(self.replacement_cost, 2),
            "currency": get("retention_roi.currency", "INR"),
            "interventions": [i.to_dict() for i in self.single_lever],
            "best_combination": self.best_combination,
            "caveat": self.caveat,
        }


class CounterfactualEngine:
    """Scores levers against a fitted attrition pipeline."""

    def __init__(self, model=None, metadata: dict | None = None) -> None:
        if model is None:
            model, metadata = load_model()
        self.model = model
        self.metadata = metadata or {}
        self.levers = load_levers()
        self.training_cost = median_training_cost()
        self.replacement_multiple = float(
            get("retention_roi.replacement_cost_monthly_multiple", 6.0)
        )
        self.max_levers = int(get("retention_roi.max_levers_per_employee", 2))

    def _risk(self, frame: pd.DataFrame) -> float:
        return float(self.model.predict_proba(frame)[:, 1][0])

    def plan_for(self, row: pd.DataFrame, recommended_course: str | None = None) -> EmployeePlan:
        """Rank every applicable lever for one employee by return on investment."""
        from hrai.ml.evaluate import risk_band

        series = row.iloc[0]
        baseline = self._risk(row)
        monthly_income = float(series.get("MonthlyIncome", 50_000) or 50_000)
        replacement_cost = monthly_income * self.replacement_multiple

        interventions: list[Intervention] = []
        for lever in self.levers:
            if not lever.applies_to(series):
                continue
            modified = row.copy()
            modified.iloc[0] = lever.apply(series)
            new_risk = self._risk(modified)
            reduction = baseline - new_risk
            cost = max(lever.cost(monthly_income, self.training_cost), 1.0)
            value = reduction * replacement_cost
            interventions.append(
                Intervention(
                    lever=lever.name,
                    label=lever.label,
                    feature=lever.feature,
                    from_value=_jsonable(series[lever.feature]),
                    to_value=_jsonable(modified.iloc[0][lever.feature]),
                    baseline_risk=baseline,
                    new_risk=new_risk,
                    risk_reduction=reduction,
                    cost=cost,
                    expected_value_saved=value,
                    roi=value / cost,
                    recommended_course=(
                        recommended_course if lever.name == "targeted_training" else None
                    ),
                )
            )

        interventions.sort(key=lambda i: -i.roi)

        best_combo = None
        applicable = [lever for lever in self.levers if lever.applies_to(series)]
        if len(applicable) >= 2 and self.max_levers >= 2:
            best_value = None
            for pair in combinations(applicable, 2):
                modified = row.copy()
                updated = series
                for lever in pair:
                    updated = lever.apply(updated)
                modified.iloc[0] = updated
                new_risk = self._risk(modified)
                reduction = baseline - new_risk
                cost = sum(
                    max(lever.cost(monthly_income, self.training_cost), 1.0) for lever in pair
                )
                value = reduction * replacement_cost
                roi = value / cost
                if best_value is None or roi > best_value["roi"]:
                    best_value = {
                        "levers": [lever.name for lever in pair],
                        "labels": [lever.label for lever in pair],
                        "baseline_risk": round(baseline, 4),
                        "new_risk": round(new_risk, 4),
                        "risk_reduction": round(reduction, 4),
                        "cost": round(cost, 2),
                        "expected_value_saved": round(value, 2),
                        "roi": round(roi, 4),
                        "note": (
                            "Combined levers assume features move independently, "
                            "which is a stronger assumption than the single-lever "
                            "estimates above."
                        ),
                    }
            best_combo = best_value

        return EmployeePlan(
            employee_id=int(series.get("employee_id", -1)),
            person_key=f"A-{int(series.get('employee_id', -1))}",
            baseline_risk=baseline,
            risk_band=risk_band(baseline),
            replacement_cost=replacement_cost,
            single_lever=interventions,
            best_combination=best_combo,
        )


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# --------------------------------------------------------------------------
# Budget-constrained allocation across the workforce
# --------------------------------------------------------------------------


def build_action_plan(
    budget: float,
    *,
    engine: CounterfactualEngine | None = None,
    candidates: pd.DataFrame | None = None,
    min_risk: float = 0.30,
    max_employees: int = 400,
    one_intervention_per_person: bool = True,
) -> dict[str, Any]:
    """Choose the interventions that maximise expected value within a budget.

    A 0/1 knapsack, solved greedily by ROI. Greedy is optimal here in the way
    that matters: intervention costs are tiny relative to any realistic budget,
    so the classic greedy failure mode — one huge item crowding out several
    small ones — does not arise. Exact dynamic programming over thousands of
    employees would cost far more compute for a difference below the noise floor
    of the risk estimates themselves.
    """
    from hrai.utils.io import load_processed

    engine = engine or CounterfactualEngine()
    if candidates is None:
        candidates = load_processed("employee_attrition_processed")

    features = candidates.drop(columns=["attrition_flag"], errors="ignore")
    probabilities = engine.model.predict_proba(features)[:, 1]
    scored = candidates.assign(_risk=probabilities)

    # Only employees actually at risk are worth spending on.
    at_risk = (
        scored[scored["_risk"] >= min_risk]
        .sort_values("_risk", ascending=False)
        .head(max_employees)
    )
    log.info(
        "action plan candidates selected",
        extra={
            "workforce": len(scored),
            "at_risk": len(at_risk),
            "min_risk": min_risk,
            "budget": budget,
        },
    )

    options: list[dict[str, Any]] = []
    for i in range(len(at_risk)):
        row = features.loc[[at_risk.index[i]]]
        plan = engine.plan_for(row)
        for intervention in plan.single_lever:
            if intervention.risk_reduction <= 0:
                continue  # never propose a lever that raises risk
            options.append(
                {
                    "employee_id": plan.employee_id,
                    "person_key": plan.person_key,
                    "baseline_risk": round(plan.baseline_risk, 4),
                    "risk_band": plan.risk_band,
                    **intervention.to_dict(),
                }
            )

    options.sort(key=lambda o: -o["roi"])

    selected: list[dict[str, Any]] = []
    spent = 0.0
    chosen_people: set[str] = set()
    for option in options:
        if one_intervention_per_person and option["person_key"] in chosen_people:
            continue
        if spent + option["cost"] > budget:
            continue
        selected.append(option)
        chosen_people.add(option["person_key"])
        spent += option["cost"]

    total_value = sum(o["expected_value_saved"] for o in selected)
    total_risk_reduction = sum(o["risk_reduction"] for o in selected)

    result = {
        "budget": round(float(budget), 2),
        "currency": get("retention_roi.currency", "INR"),
        "spend": round(spent, 2),
        "budget_utilisation": round(spent / budget, 4) if budget else 0.0,
        "employees_covered": len(chosen_people),
        "employees_at_risk": int(len(at_risk)),
        "interventions": selected,
        "expected_value_retained": round(total_value, 2),
        "expected_attritions_prevented": round(total_risk_reduction, 2),
        "return_on_investment": round(total_value / spent, 2) if spent else 0.0,
        "unfunded_at_risk": int(len(at_risk) - len(chosen_people)),
        "assumptions": {
            "replacement_cost": (
                f"{get('retention_roi.replacement_cost_monthly_multiple', 6.0)}x monthly salary"
            ),
            "min_risk_to_qualify": min_risk,
            "one_intervention_per_person": one_intervention_per_person,
            "allocation": "greedy by ROI (0/1 knapsack)",
        },
        "caveat": CAVEAT,
    }
    log.info(
        "action plan built",
        extra={
            "budget": result["budget"],
            "spend": result["spend"],
            "employees_covered": result["employees_covered"],
            "expected_value_retained": result["expected_value_retained"],
            "roi": result["return_on_investment"],
        },
    )
    return result
