"""AI Workforce Intelligence Platform — Streamlit dashboard (Step 23).

The frontend talks to the **API**, never to the services directly. That keeps
the separation honest: if the dashboard can only see what the API exposes, the
API cannot silently drift away from being the real contract.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("HRAI_API_URL", "http://localhost:8000")
API = f"{API_URL}/api/v1"

st.set_page_config(
    page_title="AI Workforce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {"HIGH": "#AC3A28", "MEDIUM": "#8A5D14", "LOW": "#316B51", "UNAVAILABLE": "#8B95A6"}


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, **params) -> dict:
    response = requests.get(f"{API}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def api_available() -> tuple[bool, dict]:
    try:
        return True, requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


def main() -> None:
    st.title("AI Workforce Intelligence Platform")

    online, health = api_available()
    if not online:
        st.error(
            f"Cannot reach the API at `{API_URL}`.\n\n"
            "Start it with `make api`, or set `HRAI_API_URL` if it runs elsewhere."
        )
        st.caption(f"Details: {health.get('error')}")
        return
    if not health.get("model_loaded"):
        st.warning("The API is up but no model is loaded. Run `make train`.")

    with st.sidebar:
        st.header("Filters")
        departments = ["All", *api_get("/dashboard/departments")["departments"]]
        department = st.selectbox("Department", departments)
        st.divider()
        st.caption(f"Model version **{health.get('model_version', 'n/a')}**")
        st.caption(f"API `{API_URL}`")

    selected = None if department == "All" else department
    summary = api_get("/dashboard/summary", department=selected or "")

    # ---- KPI row ---------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employees", f"{summary['total_employees']:,}")
    c2.metric(
        "High risk",
        f"{summary['high_risk_employees']:,}",
        help="Calibrated probability at or above the HIGH band threshold.",
    )
    c3.metric("Medium risk", f"{summary['medium_risk_employees']:,}")
    c4.metric(
        "Avg engagement",
        f"{summary['average_engagement']}/5" if summary["average_engagement"] else "—",
        help="1-5 Likert scale, Population B only.",
    )
    c5.metric("With skill gaps", f"{summary['employees_with_skill_gaps']:,}")

    st.info(
        f"**Two workforces, not one.** {summary['population_a']:,} employees come from "
        f"`employee_attrition` and {summary['population_b']:,} from "
        f"`hr_performance_engagement`. These are different companies — their overlapping "
        "employee IDs agree on gender only 48.6% of the time — so they are never joined "
        "on employee identity. They are connected through the O*NET role and skill "
        "ontology instead.",
        icon="ℹ️",
    )

    tabs = st.tabs(
        [
            "Attrition risk",
            "Engagement",
            "Skill gaps",
            "Recommendations",
            "Retention Planner",
            "Employee 360",
            "Model quality",
        ]
    )

    with tabs[0]:
        render_attrition()
    with tabs[1]:
        render_engagement()
    with tabs[2]:
        render_skill_gaps()
    with tabs[3]:
        render_recommendations(selected)
    with tabs[4]:
        render_retention_planner()
    with tabs[5]:
        render_employee_360()
    with tabs[6]:
        render_model_quality()


def render_attrition() -> None:
    st.subheader("Attrition risk by department")
    payload = api_get("/dashboard/attrition-by-department")
    df = pd.DataFrame(payload["departments"])
    if df.empty:
        st.warning("No scored employees available.")
        return

    left, right = st.columns([3, 2])
    with left:
        melted = df.melt(
            id_vars="department",
            value_vars=["average_risk", "actual_attrition_rate"],
            var_name="measure",
            value_name="rate",
        )
        melted["measure"] = melted["measure"].map(
            {"average_risk": "Predicted", "actual_attrition_rate": "Actual"}
        )
        fig = px.bar(
            melted,
            x="department",
            y="rate",
            color="measure",
            barmode="group",
            color_discrete_map={"Predicted": "#2F5D8A", "Actual": "#8B95A6"},
            labels={"rate": "Attrition rate", "department": ""},
        )
        fig.update_layout(height=380, legend_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Predicted vs actual side by side is the honest view: if the bars diverge, "
            "the model is miscalibrated for that department and you should know before "
            "acting on it."
        )
    with right:
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(payload["note"])


def render_engagement() -> None:
    st.subheader("Engagement by department")
    payload = api_get("/dashboard/engagement-by-department")
    df = pd.DataFrame(payload["departments"])
    if df.empty:
        st.warning("No engagement data available.")
        return
    fig = px.bar(
        df,
        x="average_engagement",
        y="department",
        orientation="h",
        color="average_engagement",
        color_continuous_scale="Blues",
        labels={"average_engagement": "Average engagement (1-5)", "department": ""},
    )
    fig.update_layout(height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Scale: {payload['scale']}. Population B only.")


def render_skill_gaps() -> None:
    st.subheader("Critical organisation-wide skill gaps")
    payload = api_get("/dashboard/skill-gaps", limit=20)
    st.warning(payload["note"], icon="⚠️")
    df = pd.DataFrame(payload["gaps"])
    if df.empty:
        st.info("No skill gaps computed yet.")
        return
    df["pct_of_workforce"] = (df["pct_of_workforce"] * 100).round(1)
    fig = px.bar(
        df.head(15),
        x="employees_missing",
        y="skill_name",
        orientation="h",
        color="severity_band",
        color_discrete_map={
            "HIGH": PALETTE["HIGH"],
            "MEDIUM": PALETTE["MEDIUM"],
            "LOW": PALETTE["LOW"],
        },
        hover_data=["tier", "pct_of_workforce"],
        labels={"employees_missing": "Employees missing this skill", "skill_name": ""},
    )
    fig.update_layout(
        height=520, yaxis={"categoryorder": "total ascending"}, legend_title="Severity"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_recommendations(department: str | None) -> None:
    st.subheader("AI upskilling recommendations")
    payload = api_get("/dashboard/recommendations", limit=200, department=department or "")
    df = pd.DataFrame(payload["recommendations"])
    if df.empty:
        st.info("No recommendations available.")
        return
    st.caption(
        "Each employee's highest-severity gap is matched to a course. Exact matches use "
        "explicit rules; the rest use sentence-transformer similarity, so a gap can reach "
        "a course whose title shares no words with it."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_retention_planner() -> None:
    st.subheader("Retention Planner")
    st.caption(
        "Predicting who will leave is the easy half. This answers the question an HR "
        "director actually has: given a budget, who do we spend it on, on what, and what "
        "do we get back?"
    )
    left, right = st.columns([1, 3])
    with left:
        budget = st.slider("Retention budget", 50_000, 2_000_000, 500_000, 50_000)
        min_risk = st.slider("Minimum risk to qualify", 0.10, 0.90, 0.30, 0.05)
        run = st.button("Build action plan", type="primary", use_container_width=True)

    if not run:
        st.info("Set a budget and press **Build action plan**.")
        return

    with st.spinner("Evaluating interventions for every at-risk employee..."):
        plan = api_get(
            "/intelligence/action-plan", budget=budget, min_risk=min_risk, max_employees=250
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Employees funded",
        f"{plan['employees_covered']:,}",
        delta=f"{plan['unfunded_at_risk']} unfunded",
        delta_color="inverse",
    )
    k2.metric(
        "Budget used", f"{plan['spend']:,.0f}", delta=f"{plan['budget_utilisation']:.0%} of budget"
    )
    k3.metric("Expected value retained", f"{plan['expected_value_retained']:,.0f}")
    k4.metric(
        "Return on investment",
        f"{plan['return_on_investment']}x",
        delta=f"~{plan['expected_attritions_prevented']:.0f} exits prevented",
    )

    df = pd.DataFrame(plan["interventions"])
    if not df.empty:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["intervention", "employees"]
        fig = px.bar(
            counts,
            x="employees",
            y="intervention",
            orientation="h",
            color_discrete_sequence=["#2F5D8A"],
            labels={"intervention": ""},
        )
        fig.update_layout(height=300, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df[
                [
                    "person_key",
                    "baseline_risk",
                    "risk_band",
                    "label",
                    "from_value",
                    "to_value",
                    "new_risk",
                    "risk_reduction",
                    "cost",
                    "expected_value_saved",
                    "roi",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.error(plan["caveat"], icon="🛑")


def render_employee_360() -> None:
    st.subheader("Employee 360")
    st.caption(
        "Look up by `person_key` — `A-101` or `B-1500`. A bare numeric id is ambiguous: "
        "753 ids exist in both populations, for different people."
    )
    person_key = st.text_input("person_key", value="A-622")
    if not person_key:
        return
    try:
        record = api_get(f"/employees/{person_key}")
    except requests.HTTPError:
        st.error(f"No employee `{person_key}`. Try `A-622` or `B-1001`.")
        return

    c1, c2, c3 = st.columns(3)
    probability = record.get("attrition_probability")
    c1.metric("Attrition risk", f"{probability:.1%}" if probability is not None else "Unavailable")
    c2.metric("Risk band", record.get("risk_band", "—"))
    c3.metric(
        "Engagement", f"{record['engagement_score']}/5" if record.get("engagement_score") else "—"
    )

    if record.get("risk_unavailable_reason"):
        st.warning(record["risk_unavailable_reason"], icon="⚠️")

    st.write(f"**Role** {record.get('role')}  ·  **Department** {record.get('department')}")
    st.write(f"**Skill gaps** {record.get('skill_gaps') or 'none'}")
    st.write(f"**Recommendation** {record.get('recommendation') or 'none'}")

    if str(person_key).upper().startswith("A-"):
        with st.expander("What would reduce this person's risk?", expanded=True):
            plan = api_get(f"/intelligence/counterfactual/{person_key}")
            interventions = pd.DataFrame(plan["interventions"])
            if interventions.empty:
                st.info("No applicable interventions for this employee.")
            else:
                st.dataframe(
                    interventions[
                        [
                            "label",
                            "from_value",
                            "to_value",
                            "new_risk",
                            "risk_reduction",
                            "cost",
                            "expected_value_saved",
                            "roi",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                if plan.get("best_combination"):
                    combo = plan["best_combination"]
                    st.success(
                        f"**Best pair:** {' + '.join(combo['labels'])} — risk "
                        f"{combo['baseline_risk']:.1%} → {combo['new_risk']:.1%} "
                        f"for {combo['cost']:,.0f} (ROI {combo['roi']:.2f})"
                    )
                st.error(plan["caveat"], icon="🛑")


def render_model_quality() -> None:
    st.subheader("Model quality, fairness and transfer validation")
    payload = api_get("/dashboard/model-quality")

    st.markdown("#### Held-out test performance")
    metrics = payload.get("training", {}).get("test_calibrated", {})
    if metrics:
        cols = st.columns(6)
        for col, key in zip(
            cols, ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier"], strict=False
        ):
            col.metric(key.replace("_", " ").upper(), metrics.get(key))

    st.markdown("#### What drives attrition")
    drivers = pd.DataFrame(payload.get("global_drivers", []))
    if not drivers.empty:
        fig = px.bar(
            drivers.head(10),
            x="importance",
            y="label",
            orientation="h",
            color_discrete_sequence=["#2F5D8A"],
            labels={"label": ""},
        )
        fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Fairness audit")
    fairness = payload.get("fairness", {})
    for attribute, result in (fairness.get("attributes") or {}).items():
        verdict = "within tolerance" if result["within_tolerance"] else "FLAGGED"
        with st.expander(f"{attribute} — {verdict}", expanded=not result["within_tolerance"]):
            st.write(fairness.get("interpretation", {}).get(attribute, ""))
            st.dataframe(pd.DataFrame(result["groups"]).T, use_container_width=True)
    if fairness.get("limitation"):
        st.caption(fairness["limitation"])

    st.markdown("#### Cross-population transfer")
    transfer = payload.get("transfer_validation", {})
    if transfer:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "ROC-AUC on Population A", transfer["population_a"]["held_out_metrics"]["roc_auc"]
        )
        c2.metric(
            "ROC-AUC on Population B", transfer["population_b"]["external_metrics"]["roc_auc"]
        )
        c3.metric("Drop on transfer", transfer["roc_auc_drop_on_transfer"])
        st.error(transfer["interpretation"], icon="🛑")
        st.dataframe(
            pd.DataFrame(transfer["distribution_shift"]), use_container_width=True, hide_index=True
        )


if __name__ == "__main__":
    main()
