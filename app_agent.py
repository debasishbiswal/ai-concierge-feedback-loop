"""
AI Concierge: Planning Agent
--------------------------------

This Streamlit app extends the rubric‑based analysis by adding
agentic behaviour. In addition to analysing guest comments and
surfacing the most impactful themes, it can:

* Store the results of a run for later reference (memory).
* Generate a lightweight action plan for the top themes,
  including effort, risk, owner hints and actionable steps.
* Export the plan as Markdown, a Jira‑compatible CSV or
  open a GitHub issue (via secrets) with the plan details.
* Create calendar follow‑up events in an iCalendar (.ics) file
  so you can schedule work on each theme or step.
* Simulate the impact of fixing a single issue by applying
  adoption and effectiveness multipliers and visualising
  before/after sentiment. Two narrative voices (Impact Analyst
  and Risk Reviewer) explain the outcomes when an API key
  is available.

The file does not replace any existing app; you can deploy it
alongside your other dashboards by pointing Streamlit to
`app_agent.py` as the main file. The original `app.py` and
`app_rubric.py` remain unchanged.
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from textblob import TextBlob


# ---------------------------------------------------------------------------
# Configuration and helper functions
# ---------------------------------------------------------------------------

# Configure Streamlit page
st.set_page_config(page_title="AI Concierge: Planning Agent", layout="wide")


def get_api_key() -> str:
    """Retrieve an OpenAI API key from secrets or session state.

    If a key is stored in st.secrets["OPENAI_API_KEY"] it is used. Otherwise
    falls back to a value typed in the sidebar (stored in session state).
    Returns an empty string if no key is available.
    """
    key: str = st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        key = st.session_state.get("api_key_input", "")
    return key


def baseline_sentiment(text: str) -> float:
    """Compute a baseline sentiment for a comment using TextBlob polarity.

    The value is clipped to [-1, 1]. On any error, returns 0.0.
    """
    try:
        polarity: float = float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, polarity))


def extract_json(text: str) -> Dict[str, Any] | None:
    """Extract a JSON object from a string.

    Some LLM responses may include extra prose around a JSON object. This
    helper first tries to parse the entire string as JSON. Failing that, it
    searches for the first substring enclosed in curly braces and parses
    that. Returns None if no valid JSON can be extracted.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def clip_val(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a number to the closed interval [lo, hi]."""
    try:
        xf = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xf))


# Qualitative impact mapping used by the rubric analysis. These values
# reflect the expected sentiment improvement when a suggested fix is
# implemented.
IMPACT_SCALE: Dict[str, float] = {
    "very_low": 0.05,
    "low": 0.15,
    "medium": 0.30,
    "high": 0.50,
    "very_high": 0.80,
}

# Weight applied to the model‑provided confidence when calculating lift.
# A value of 1.0 means the impact label is fully scaled by the confidence.
CONFIDENCE_WEIGHT: float = 1.0

# System prompt instructing the model how to respond for each comment. The
# model must return only JSON with the specified keys. We avoid dashes in
# the prompt to keep theme names clean.
SYSTEM_PROMPT: str = (
    "You are a hotel-operations analyst. "
    "Return ONLY JSON with keys: "
    "theme (string), impact (one of 'very_low','low','medium','high','very_high'), "
    "suggestion (string), confidence (number 0..1), impact_reason (string). "
    "Theme must be short (2-4 words) without dashes. "
    "Pick the SINGLE best theme. "
    "impact reflects how much the guest sentiment would improve if your suggestion is implemented. "
    "Do not include any text outside the JSON."
)


def call_llm_for_comment(client, model: str, comment: str) -> str:
    """Call OpenAI's chat API for a single comment and return the content.

    Uses temperature=0 for deterministic responses. For GPT‑4 models we
    request response_format={"type": "json_object"} so the model returns
    structured JSON directly.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Feedback: {comment}"},
        ],
    }
    if model.startswith("gpt-4"):
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def analyse_with_rubric(df: pd.DataFrame, api_key: str, max_comments: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse comments using the rubric approach.

    Returns (summary_df, detail_df). summary_df contains aggregated metrics
    by theme; detail_df contains per-comment analysis results.
    """
    from openai import OpenAI

    if "comment_text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'comment_text' column.")

    client = OpenAI(api_key=api_key)
    models = ["gpt-4o", "gpt-3.5-turbo"]

    # Prepare comments: strip and limit, skip empties
    comments_series = df["comment_text"].astype(str).str.strip()
    comments_series = comments_series[comments_series != ""]
    comments: List[str] = comments_series.head(max_comments).tolist()

    records: List[Dict[str, Any]] = []
    for comment in comments:
        base = baseline_sentiment(comment)
        result_json: Dict[str, Any] | None = None
        for m in models:
            try:
                raw = call_llm_for_comment(client, m, comment)
                js = extract_json(raw)
                if js is not None:
                    result_json = js
                    break
            except Exception:
                continue
        if result_json is None:
            continue
        theme = (str(result_json.get("theme", "")).strip() or "Other")
        impact = str(result_json.get("impact", "medium")).strip().lower()
        suggestion = (str(result_json.get("suggestion", "")).strip() or "")
        confidence_raw = result_json.get("confidence", 0.7)
        reason = (str(result_json.get("impact_reason", "")).strip() or "")
        if impact not in IMPACT_SCALE:
            impact = "medium"
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.7
        confidence = clip_val(confidence, 0.0, 1.0)
        lift_raw = IMPACT_SCALE[impact]
        lift = clip_val(lift_raw * (CONFIDENCE_WEIGHT * confidence + (1.0 - CONFIDENCE_WEIGHT)))
        predicted_post = clip_val(base + lift)
        records.append({
            "theme": theme,
            "suggestion": suggestion,
            "impact_label": impact,
            "confidence": round(confidence, 3),
            "impact_reason": reason,
            "baseline_sentiment": round(base, 3),
            "predicted_post_sentiment": round(predicted_post, 3),
            "sentiment_lift": round(predicted_post - base, 3),
            "comment_text": comment,
        })
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    detail_df = pd.DataFrame.from_records(records)
    summary_df = (
        detail_df.groupby("theme", dropna=False)
        .agg(
            complaints_count=("theme", "count"),
            avg_sentiment_lift=("sentiment_lift", "mean"),
            avg_predicted_post_sentiment=("predicted_post_sentiment", "mean"),
            impact_mode=("impact_label", lambda x: x.value_counts().index[0]),
            top_suggestion=("suggestion", lambda x: x.value_counts().index[0] if len(x) else ""),
        )
        .sort_values("avg_sentiment_lift", ascending=False)
        .reset_index()
    )
    summary_df["avg_sentiment_lift"] = summary_df["avg_sentiment_lift"].round(3)
    summary_df["avg_predicted_post_sentiment"] = summary_df["avg_predicted_post_sentiment"].round(3)
    summary_df["priority_score"] = (summary_df["complaints_count"] * summary_df["avg_sentiment_lift"]).round(3)
    return summary_df, detail_df


def plan_for_theme(client, model: str, theme: str, suggestion: str, avg_lift: float) -> Dict[str, Any]:
    """Request a lightweight plan for a theme from the LLM.

    The plan includes effort (1..5), risk (1..5), owner_hint and a list of steps.
    Returns a dictionary with these fields and the computed ICE score.
    """
    prompt = (
        f"You're part of a hotel operations team. We identified a recurring theme: '{theme}'. "
        f"The current recommended fix is: '{suggestion}'. The average sentiment lift if this issue is solved is {avg_lift:.2f}. "
        "Provide a short plan in JSON with keys: effort (1..5), risk (1..5), owner_hint (string), steps (array of 3 to 6 concise steps). "
        "Effort is how much work is required, risk is the chance of negative side effects. "
        "Keep answers short; do not include any text outside the JSON."
    )
    messages = [
        {"role": "system", "content": "You are a detail‑oriented operations planner."},
        {"role": "user", "content": prompt},
    ]
    kwargs: Dict[str, Any] = {"model": model, "temperature": 0, "messages": messages}
    if model.startswith("gpt-4"):
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    plan_json = extract_json(content)
    if not plan_json:
        # Fallback: simple default plan
        plan_json = {
            "effort": 3,
            "risk": 2,
            "owner_hint": "Operations",
            "steps": ["Investigate root cause", "Draft SOP", "Train staff"],
        }
    # Validate and coerce fields
    try:
        eff = int(plan_json.get("effort", 3))
    except Exception:
        eff = 3
    eff = max(1, min(5, eff))
    try:
        risk = int(plan_json.get("risk", 3))
    except Exception:
        risk = 3
    risk = max(1, min(5, risk))
    owner_hint = str(plan_json.get("owner_hint", "Operations")).strip() or "Operations"
    steps = plan_json.get("steps", ["Investigate", "Implement", "Verify"])
    if not isinstance(steps, list):
        steps = [str(steps)]
    steps = [str(s).strip() for s in steps if str(s).strip()]
    if len(steps) < 1:
        steps = ["Identify root cause", "Fix and monitor"]
    # Compute ICE score: (lift / effort)
    ice = avg_lift * (1.0 / eff) if eff else 0.0
    return {
        "effort": eff,
        "risk": risk,
        "owner_hint": owner_hint,
        "steps": steps,
        "ice_score": round(ice, 3),
    }


def generate_plan(summary_df: pd.DataFrame, api_key: str, top_n: int = 5) -> pd.DataFrame:
    """Generate a plan for the top N themes based on priority score.

    Calls the LLM for each theme to get effort, risk, owner_hint and steps.
    Returns a DataFrame with columns: theme, effort, risk, owner_hint, steps,
    avg_lift, complaints_count, ice_score.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    models = ["gpt-4o", "gpt-3.5-turbo"]
    # Sort themes by priority_score descending
    df_sorted = summary_df.sort_values("priority_score", ascending=False).head(top_n).reset_index(drop=True)
    plans: List[Dict[str, Any]] = []
    for idx, row in df_sorted.iterrows():
        theme = str(row["theme"])
        suggestion = str(row["top_suggestion"])
        avg_lift = float(row["avg_sentiment_lift"])
        plan_data: Dict[str, Any] | None = None
        # Try models in order
        for m in models:
            try:
                plan_data = plan_for_theme(client, m, theme, suggestion, avg_lift)
                if plan_data:
                    break
            except Exception:
                continue
        if not plan_data:
            plan_data = {
                "effort": 3,
                "risk": 3,
                "owner_hint": "Operations",
                "steps": ["Investigate root cause", "Draft SOP", "Train staff"],
                "ice_score": avg_lift / 3 if avg_lift else 0.0,
            }
        plans.append({
            "theme": theme,
            "avg_lift": avg_lift,
            "complaints_count": int(row["complaints_count"]),
            "effort": plan_data["effort"],
            "risk": plan_data["risk"],
            "owner_hint": plan_data["owner_hint"],
            "steps": plan_data["steps"],
            "ice_score": plan_data["ice_score"],
        })
    plan_df = pd.DataFrame(plans)
    # Sort by ice_score descending
    plan_df = plan_df.sort_values("ice_score", ascending=False).reset_index(drop=True)
    return plan_df


def save_run_to_json(summary_df: pd.DataFrame, plan_df: pd.DataFrame, detail_df: pd.DataFrame) -> str:
    """Save the results of a run into a JSON file and return the file path."""
    runs_dir = "/mnt/data/runs"
    os.makedirs(runs_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"run_{timestamp}.json"
    filepath = os.path.join(runs_dir, filename)
    obj = {
        "meta": {
            "timestamp": timestamp,
            "num_comments": int(detail_df.shape[0]),
            "num_themes": int(summary_df.shape[0]),
        },
        "summary": summary_df.to_dict(orient="records"),
        "plan": plan_df.to_dict(orient="records"),
        "detail": detail_df.to_dict(orient="records"),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return filepath


def convert_plan_to_markdown(plan_df: pd.DataFrame) -> str:
    """Convert the plan DataFrame into a Markdown representation."""
    lines: List[str] = ["# Action Plan\n"]
    for _, row in plan_df.iterrows():
        lines.append(f"## {row['theme']}")
        lines.append(f"* Effort: {row['effort']} / 5")
        lines.append(f"* Risk: {row['risk']} / 5")
        lines.append(f"* Owner hint: {row['owner_hint']}")
        lines.append(f"* Average lift: {row['avg_lift']:.3f}")
        lines.append(f"* Complaints count: {row['complaints_count']}")
        lines.append("* Steps:")
        for step in row['steps']:
            lines.append(f"  - {step}")
        lines.append("")
    return "\n".join(lines)


def convert_plan_to_jira_csv(plan_df: pd.DataFrame) -> str:
    """Convert the plan DataFrame into a CSV string suitable for Jira import."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    # Headers for Jira: Summary, Description
    writer.writerow(["Summary", "Description"])
    for _, row in plan_df.iterrows():
        summary = f"Improve {row['theme']}".strip()
        desc_lines = [
            f"Effort: {row['effort']} / 5",
            f"Risk: {row['risk']} / 5",
            f"Owner hint: {row['owner_hint']}",
            f"Average lift: {row['avg_lift']:.3f}",
            f"Complaints count: {row['complaints_count']}",
            "Steps:",
        ]
        for step in row['steps']:
            desc_lines.append(f"- {step}")
        description = "\n".join(desc_lines)
        writer.writerow([summary, description])
    return output.getvalue()


def generate_ics_events(plan_df: pd.DataFrame, start_date: datetime, days_between: int, style: str = "per_theme") -> str:
    """Generate a simple iCalendar (.ics) string for follow-up events.

    style can be 'per_theme' (one event per theme) or 'per_step' (one event per step).
    The events are scheduled starting from start_date with days_between days between them.
    """
    events: List[str] = []
    dt_format = "%Y%m%dT%H%M%S"
    seq = 0
    current = start_date
    for _, row in plan_df.iterrows():
        theme = row["theme"]
        if style == "per_theme":
            summary = f"Guest Exp: {theme}"
            begin = current.strftime(dt_format)
            end_dt = current + timedelta(hours=1)
            end = end_dt.strftime(dt_format)
            uid = f"{seq}@aiconcierge"
            events.append(
                "BEGIN:VEVENT\n"
                f"UID:{uid}\n"
                f"DTSTAMP:{datetime.utcnow().strftime(dt_format)}Z\n"
                f"DTSTART:{begin}Z\n"
                f"DTEND:{end}Z\n"
                f"SUMMARY:{summary}\n"
                "END:VEVENT\n"
            )
            current = current + timedelta(days=days_between)
            seq += 1
        else:  # per_step
            for step in row["steps"]:
                summary = f"Guest Exp: {theme} - {step}"
                begin = current.strftime(dt_format)
                end_dt = current + timedelta(hours=1)
                end = end_dt.strftime(dt_format)
                uid = f"{seq}@aiconcierge"
                events.append(
                    "BEGIN:VEVENT\n"
                    f"UID:{uid}\n"
                    f"DTSTAMP:{datetime.utcnow().strftime(dt_format)}Z\n"
                    f"DTSTART:{begin}Z\n"
                    f"DTEND:{end}Z\n"
                    f"SUMMARY:{summary}\n"
                    "END:VEVENT\n"
                )
                current = current + timedelta(days=days_between)
                seq += 1
    calendar = "BEGIN:VCALENDAR\nVERSION:2.0\nCALSCALE:GREGORIAN\n" + "".join(events) + "END:VCALENDAR"
    return calendar


def simulate_theme_impact(detail_df: pd.DataFrame, theme: str, adoption: float, effectiveness: float) -> Tuple[float, float, float, float]:
    """Simulate the impact of fixing a theme.

    adoption and effectiveness are fractions between 0 and 1. The new predicted
    sentiment for the theme is computed as: baseline + (lift * adoption *
    effectiveness). Returns a tuple of (old_avg_baseline, old_avg_predicted,
    new_avg_predicted_theme, new_avg_predicted_overall).
    """
    if detail_df.empty:
        return 0.0, 0.0, 0.0, 0.0
    # Baseline and predicted across all comments
    overall_baseline = detail_df["baseline_sentiment"].mean()
    overall_predicted = detail_df["predicted_post_sentiment"].mean()
    # Filter by theme
    theme_df = detail_df[detail_df["theme"] == theme]
    if theme_df.empty:
        return overall_baseline, overall_predicted, overall_predicted, overall_predicted
    old_theme_predicted = theme_df["predicted_post_sentiment"].mean()
    # New predicted for theme: baseline + lift * adoption * effectiveness
    new_pred_values = theme_df["baseline_sentiment"] + theme_df["sentiment_lift"] * adoption * effectiveness
    new_pred_values = new_pred_values.clip(-1.0, 1.0)
    new_theme_predicted = new_pred_values.mean()
    # Compute new overall predicted by replacing old theme predicted values
    # Compute aggregated predicted for non-theme comments
    non_theme_df = detail_df[detail_df["theme"] != theme]
    if non_theme_df.empty:
        new_overall = new_theme_predicted
    else:
        new_overall = (
            new_pred_values.sum() + non_theme_df["predicted_post_sentiment"].sum()
        ) / detail_df.shape[0]
    return overall_baseline, overall_predicted, new_theme_predicted, new_overall


def generate_narratives(theme: str, adoption: float, effectiveness: float, summary_df: pd.DataFrame, detail_df: pd.DataFrame, api_key: str) -> Tuple[str, str]:
    """Generate Impact Analyst and Risk Reviewer narratives for a simulation.

    Returns (impact_story, risk_story). If API key is missing or calls fail,
    returns empty strings.
    """
    from openai import OpenAI
    if not api_key:
        return "", ""
    client = OpenAI(api_key=api_key)
    models = ["gpt-4o", "gpt-3.5-turbo"]
    # Build a concise data description for the model
    theme_row = summary_df[summary_df["theme"] == theme]
    if theme_row.empty:
        return "", ""
    row = theme_row.iloc[0]
    desc = (
        f"Theme: {theme}, Complaints: {row['complaints_count']}, Avg lift: {row['avg_sentiment_lift']:.3f}, "
        f"Adoption rate: {adoption:.2f}, Effectiveness: {effectiveness:.2f}."
    )
    impact_prompt = (
        "You are an Impact Analyst. "
        "Given the following context, write a short, optimistic explanation (2-3 sentences) of the expected benefits if this theme is addressed. "
        "Be specific about how the fixes will help guests and business metrics. Context: " + desc
    )
    risk_prompt = (
        "You are a Risk Reviewer. "
        "Given the following context, write a short, cautious explanation (2-3 sentences) of potential downsides or challenges if this theme is addressed. "
        "Highlight any trade-offs. Context: " + desc
    )
    impact_story, risk_story = "", ""
    for m in models:
        try:
            resp = client.chat.completions.create(model=m, temperature=0, messages=[
                {"role": "system", "content": "You are concise and helpful."},
                {"role": "user", "content": impact_prompt},
            ])
            impact_story = resp.choices[0].message.content.strip()
            break
        except Exception:
            continue
    for m in models:
        try:
            resp = client.chat.completions.create(model=m, temperature=0, messages=[
                {"role": "system", "content": "You are concise and critical."},
                {"role": "user", "content": risk_prompt},
            ])
            risk_story = resp.choices[0].message.content.strip()
            break
        except Exception:
            continue
    return impact_story, risk_story


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("AI Concierge: Planning Agent")
st.markdown(
    "This dashboard uses a team of small AI agents to turn guest feedback into action. "
    "We begin by measuring each comment's baseline mood and how much happier the guest might be once we fix the issue. "
    "From those predictions we highlight themes that offer the biggest bang for your buck. "
    "Then our agents draft plans, schedule follow‑ups and simulate the impact of your decisions so you can prioritise confidently."
)

# Data upload
st.header("1) Data")
uploaded_file = st.file_uploader("Upload a CSV containing a 'comment_text' column", type=["csv"])

data_df: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        data_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(data_df):,} rows from your file.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        data_df = None

if data_df is None:
    st.info("No valid file uploaded. Using a small built‑in sample.")
    sample = pd.DataFrame({
        "comment_text": [
            "Wi‑Fi keeps dropping in my room and the lobby.",
            "Parking fees are too high for overnight guests.",
            "Check‑in took forever; only one agent at the desk.",
            "Gym is great but opens too late for business trips.",
            "Pool hours end too early; kids were disappointed.",
            "Air conditioning was noisy at night.",
            "Room was spotless and staff were friendly.",
            "Breakfast buffet had long lines and ran out of eggs.",
        ]
    })
    data_df = sample.copy()


# Determine available filters
property_col: str | None = None
if data_df is not None:
    for col_name in ["property", "property_name", "property_id", "hotel", "property_code"]:
        if col_name in data_df.columns:
            property_col = col_name
            break

# Sidebar filters
st.sidebar.header("Filters")
filtered_df: pd.DataFrame = data_df.copy()
if property_col:
    props = sorted(filtered_df[property_col].dropna().astype(str).unique())
    selected_props = st.sidebar.multiselect("Select properties", options=props, default=props)
    filtered_df = filtered_df[filtered_df[property_col].astype(str).isin(selected_props)].copy()

# Additional attributes for filtering
if not filtered_df.empty:
    possible_cols: Dict[str, List[str]] = {
        "City": ["city", "location_city", "guest_city"],
        "State": ["state", "location_state"],
        "Property type": ["property_type", "property_class", "hotel_type"],
        "Channel": ["channel", "booking_channel", "feedback_channel"],
        "Stay purpose": ["stay_purpose", "trip_purpose"],
        "Loyalty tier": ["loyalty_tier", "membership_level"],
    }
    for label, candidates in possible_cols.items():
        col = None
        for cand in candidates:
            if cand in filtered_df.columns:
                col = cand
                break
        if col:
            options = sorted(filtered_df[col].dropna().astype(str).unique())
            if options:
                selected = st.sidebar.multiselect(label, options=options, default=options)
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)].copy()


# Data preview
st.subheader("Data being analysed")
if not filtered_df.empty:
    st.dataframe(filtered_df.head(1000), use_container_width=True)
else:
    st.write("No data available for analysis.")


# Sidebar settings
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("OpenAI API key", type="password", placeholder="sk-...", key="api_key_input")
key_source = "secrets" if st.secrets.get("OPENAI_API_KEY") else ("input" if api_key_input else "missing")
st.sidebar.write("Key source:", key_source)
max_comments = st.sidebar.slider("Max comments to analyse", min_value=20, max_value=500, value=120, step=20)


# Analysis
st.header("2) AI-driven analysis")
st.markdown(
    "We measure each comment's starting mood and ask the model to predict how much happier the guest would be once a fix is in place. "
    "By aggregating these per-comment predictions, we surface the themes with the greatest potential impact."
)
if st.button("Run analysis"):
    api_key = get_api_key()
    if not api_key:
        st.error("Please provide an OpenAI API key in the sidebar or via Streamlit secrets.")
    else:
        with st.spinner("Running analysis..."):
            summary_df, detail_df = analyse_with_rubric(filtered_df, api_key, max_comments=max_comments)
        if summary_df.empty:
            st.error("The analysis returned no results. Try a different dataset or adjust filters.")
        else:
            st.success("Analysis complete.")
            st.subheader("Theme summary")
            st.dataframe(summary_df, use_container_width=True)
            # Donut chart for complaints count
            top_themes = summary_df.sort_values("complaints_count", ascending=False).head(10)
            if not top_themes.empty:
                pie = (
                    alt.Chart(top_themes)
                    .mark_arc(innerRadius=50, outerRadius=100)
                    .encode(
                        theta=alt.Theta("complaints_count:Q", stack=True),
                        color=alt.Color("theme:N", legend=None),
                        tooltip=["theme:N", "complaints_count:Q"]
                    )
                )
                st.altair_chart(pie, use_container_width=True)
                st.markdown("""The donut chart shows which themes receive the most feedback. Larger slices mean more guest comments.")
            # Lift distribution density
            density_df = detail_df[["sentiment_lift"]].copy()
            density_df["sentiment_lift"] = density_df["sentiment_lift"].astype(float)
            hist = (
                alt.Chart(density_df)
                .transform_density(
                    "sentiment_lift",
                    as_=["sentiment_lift", "density"],
                    bandwidth=0.05,
                )
                .mark_area(opacity=0.6)
                .encode(
                    x=alt.X("sentiment_lift:Q", title="Predicted lift (per comment)"),
                    y=alt.Y("density:Q", title="Density"),
                )
            )
            # Add mean and median lines
            mean_lift = float(density_df["sentiment_lift"].mean()) if not density_df.empty else 0.0
            median_lift = float(density_df["sentiment_lift"].median()) if not density_df.empty else 0.0
            mean_line = alt.Chart(pd.DataFrame({"sentiment_lift": [mean_lift]})).mark_rule(color="red").encode(x="sentiment_lift:Q")
            median_line = alt.Chart(pd.DataFrame({"sentiment_lift": [median_lift]})).mark_rule(color="blue").encode(x="sentiment_lift:Q")
            st.altair_chart(hist + mean_line + median_line, use_container_width=True)
            st.markdown("""This curve shows how predicted lifts are distributed. Peaks indicate where most comments cluster. The red line marks the mean lift and the blue line marks the median.""")
            # Scatter plot baseline vs predicted
            scatter_data = detail_df[["baseline_sentiment", "predicted_post_sentiment", "impact_label"]].copy()
            scatter = (
                alt.Chart(scatter_data)
                .mark_circle(size=60, opacity=0.6)
                .encode(
                    x=alt.X("baseline_sentiment:Q", title="Baseline sentiment"),
                    y=alt.Y("predicted_post_sentiment:Q", title="Predicted post sentiment"),
                    color=alt.Color("impact_label:N", title="Impact label"),
                    tooltip=["baseline_sentiment:Q", "predicted_post_sentiment:Q", "impact_label:N"],
                )
            )
            identity_line = alt.Chart(pd.DataFrame({"x": [-1, 1], "y": [-1, 1]})).mark_line(strokeDash=[5, 5], color="gray").encode(x="x:Q", y="y:Q")
            st.altair_chart(scatter + identity_line, use_container_width=True)
            st.markdown("""Each point compares a comment's baseline mood to its predicted post‑fix mood. Points above the grey line indicate an improvement.""")
            # Memory, planning and actions
            st.header("3) Memory & planning")
            st.markdown(
                "Save your analysis, generate plans for the top themes, schedule follow‑ups and create issues in your task tracker."
            )
            # Generate plan button
            with st.spinner("Preparing plan..."):
                plan_df = generate_plan(summary_df, api_key, top_n=5)
            st.subheader("Action plan")
            st.dataframe(plan_df, use_container_width=True)
            st.markdown(
                "Plans include an effort and risk score (1=low, 5=high), an owner hint and a list of steps. ICE score = lift ÷ effort."
            )
            # Show plan bar chart
            if not plan_df.empty:
                bar = (
                    alt.Chart(plan_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("ice_score:Q", title="ICE score"),
                        y=alt.Y("theme:N", sort="-x", title="Theme"),
                        tooltip=["theme:N", "ice_score:Q", "effort:Q", "risk:Q"],
                        color=alt.Color("effort:Q", title="Effort", scale=alt.Scale(scheme="blues"))
                    )
                )
                st.altair_chart(bar, use_container_width=True)
                st.markdown("""Longer bars mean higher priority (big lift and low effort). Darker colours indicate higher effort.""")
            # Export buttons
            st.subheader("Memory and actions")
            run_name = st.text_input("Name this run", value="My run")
            if st.button("Save run to JSON"):
                path = save_run_to_json(summary_df, plan_df, detail_df)
                st.success(f"Run saved to {path}")
                with open(path, "rb") as f:
                    st.download_button("Download run JSON", f, file_name=os.path.basename(path))
            if st.button("Download plan as Markdown"):
                md = convert_plan_to_markdown(plan_df)
                st.download_button("Download Markdown", md, file_name=f"{run_name.replace(' ','_')}_plan.md")
            if st.button("Download plan as Jira CSV"):
                csv_str = convert_plan_to_jira_csv(plan_df)
                st.download_button("Download Jira CSV", csv_str, file_name=f"{run_name.replace(' ','_')}_plan.csv")
            # Calendar follow‑ups
            st.subheader("Calendar follow‑ups")
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("Start date", value=datetime.utcnow().date())
            with col2:
                spacing = st.number_input("Days between tasks", min_value=1, max_value=30, value=7)
            style = st.selectbox("Event style", options=["per_theme", "per_step"], format_func=lambda x: "One event per theme" if x == "per_theme" else "One event per step")
            if st.button("Download calendar (.ics)"):
                ics = generate_ics_events(plan_df, datetime.combine(start, datetime.min.time()), int(spacing), style=style)
                st.download_button("Download .ics", ics, file_name=f"{run_name.replace(' ','_')}_calendar.ics")
            # Simulation
            st.header("4) Simulation")
            st.markdown(
                "Pick a theme from your plan to see how fixing it could change overall sentiment. Set adoption and effectiveness and compare before and after."
            )
            if not plan_df.empty:
                sim_theme = st.selectbox("Theme to simulate", options=plan_df["theme"].tolist())
                adoption_pct = st.slider("Adoption rate (%)", min_value=0, max_value=100, value=70, step=5)
                effectiveness_pct = st.slider("Effectiveness (%)", min_value=0, max_value=100, value=80, step=5)
                if st.button("Run simulation"):
                    adoption = adoption_pct / 100.0
                    effectiveness = effectiveness_pct / 100.0
                    old_baseline, old_predicted, new_theme_pred, new_overall_pred = simulate_theme_impact(
                        detail_df, sim_theme, adoption, effectiveness
                    )
                    # Show metrics
                    st.write("**Overall metrics**")
                    st.table(
                        pd.DataFrame({
                            "Metric": ["Overall baseline", "Overall predicted", "New overall predicted"],
                            "Value": [round(old_baseline, 3), round(old_predicted, 3), round(new_overall_pred, 3)],
                        })
                    )
                    # Bar chart for the chosen theme
                    sim_df = pd.DataFrame({
                        "Scenario": ["Before", "After"],
                        "Average sentiment": [old_predicted, new_overall_pred],
                    })
                    sim_bar = (
                        alt.Chart(sim_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Scenario:N"),
                            y=alt.Y("Average sentiment:Q", scale=alt.Scale(domain=[-1, 1])),
                            color=alt.Color("Scenario:N", legend=None),
                            tooltip=["Scenario:N", "Average sentiment:Q"]
                        )
                    )
                    st.altair_chart(sim_bar, use_container_width=True)
                    # Explain the simulation bar chart in plain language
                    st.markdown(
                        "The bar chart compares the average predicted sentiment before and after fixing the selected theme. "
                        "A taller bar in the 'After' column means that, on average, guests would feel better once the fix is in place."
                    )
                    # Narratives
                    impact_story, risk_story = generate_narratives(
                        sim_theme, adoption, effectiveness, summary_df, detail_df, api_key
                    )
                    if impact_story:
                        st.write("**Impact Analyst:**", impact_story)
                    if risk_story:
                        st.write("**Risk Reviewer:**", risk_story)