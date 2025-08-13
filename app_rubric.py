"""
Streamlit application implementing a rubric-based analysis for guest feedback.

This version extends the original AI Concierge app by calculating a baseline
sentiment for each comment using TextBlob, asking an LLM to classify the
impact of a suggested fix into qualitative categories, mapping those
categories to numeric lift values, and deriving a predicted post‑fix sentiment.

Metrics are aggregated by theme and visualised through several charts. A
detailed explanation of each metric is provided in the UI.

To deploy alongside the existing app, simply add this file to your
repository and point Streamlit to `app_rubric.py` as the main file for a
second app. It does not modify or replace the original app.
"""

import os
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from textblob import TextBlob

__all__ = []  # explicit export list

# -----------------------------------------------------------------------------
# Configuration and helper functions
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AI Concierge — Rubric Agent", layout="wide")

def get_api_key() -> str:
    """Retrieve an OpenAI API key either from Streamlit secrets or the session.

    This helper prefers the value stored in st.secrets["OPENAI_API_KEY"] but
    falls back to a session‑state text input if not present. Returns an empty
    string if no key is available.
    """
    key: str = st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        key = st.session_state.get("api_key_input", "")
    return key


def baseline_sentiment(text: str) -> float:
    """Compute a baseline sentiment score using TextBlob polarity.

    The value is clipped to the range [-1, 1] to ensure consistency.
    If any exception occurs, returns 0.0.
    """
    try:
        polarity: float = float(TextBlob(text).sentiment.polarity)
        return max(-1.0, min(1.0, polarity))
    except Exception:
        return 0.0


def extract_json(text: str) -> dict | None:
    """Attempt to extract a JSON object from the given text.

    Some LLM responses may include additional prose around the JSON. This
    function first tries to parse the entire string, and if that fails,
    searches for the first substring enclosed in curly braces and attempts to
    parse that. Returns None if no JSON can be extracted.
    """
    if not text:
        return None
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # best‑effort: extract first {...}
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def clip_val(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a numeric value to the interval [lo, hi]."""
    try:
        xf = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, xf))


# Qualitative impact mapping to numeric lifts. These values reflect how much
# guest sentiment is expected to improve when the suggested fix is applied.
IMPACT_SCALE: dict[str, float] = {
    "very_low": 0.05,
    "low": 0.15,
    "medium": 0.30,
    "high": 0.50,
    "very_high": 0.80,
}

# Weight applied to the model‑provided confidence when calculating lift. A
# value of 1.0 means the impact label is fully scaled by the confidence. A
# value of 0.0 ignores confidence altogether.
CONFIDENCE_WEIGHT: float = 1.0

# System prompt to instruct the LLM how to respond. The model must return
# pure JSON with specific keys: theme, impact, suggestion, confidence and
# impact_reason.
SYSTEM_PROMPT: str = (
    "You are a hotel‑operations analyst. "
    "Return ONLY JSON with keys: "
    "theme (string), impact (one of 'very_low','low','medium','high','very_high'), "
    "suggestion (string), confidence (number 0..1), impact_reason (string). "
    "Theme must be short (2–4 words). "
    "Pick the SINGLE best theme. "
    "impact reflects how much the guest sentiment would improve if your suggestion is implemented. "
    "Do not include any text outside the JSON."
)


def call_llm_for_comment(client, model: str, comment: str) -> str:
    """Call the chat completion API for a single comment and return the content.

    Uses a temperature of 0 for determinism. For GPT‑4 models we request
    response_format = {"type": "json_object"} so that the model returns only JSON.
    """
    kwargs: dict = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Feedback: {comment}"},
        ],
    }
    # GPT‑4 family supports structured JSON responses
    if model.startswith("gpt-4"):
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def analyse_with_rubric(df: pd.DataFrame, api_key: str, max_comments: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse a DataFrame of comments using the rubric approach.

    Returns a tuple of (theme_summary_df, detail_df). The first contains
    aggregated metrics by theme; the second contains per‑comment details.
    """
    from openai import OpenAI  # imported lazily to avoid heavy import if not used

    if "comment_text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'comment_text' column.")

    client = OpenAI(api_key=api_key)
    models = ["gpt-4o", "gpt-3.5-turbo"]

    # Prepare comments: strip and limit to max_comments, skip empty strings
    comments_series = df["comment_text"].astype(str).str.strip()
    comments_series = comments_series[comments_series != ""]
    comments = comments_series.head(max_comments).tolist()

    records: list[dict] = []
    for comment in comments:
        base = baseline_sentiment(comment)
        result_json: dict | None = None
        # Attempt multiple models in order until one succeeds
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
        # Validate impact category
        if impact not in IMPACT_SCALE:
            impact = "medium"
        # Parse confidence
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.7
        confidence = clip_val(confidence, 0.0, 1.0)
        # Convert qualitative impact to numeric lift
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

    # Aggregate by theme
    theme_summary = (
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
    # Round numeric columns
    theme_summary["avg_sentiment_lift"] = theme_summary["avg_sentiment_lift"].round(3)
    theme_summary["avg_predicted_post_sentiment"] = theme_summary["avg_predicted_post_sentiment"].round(3)
    # Compute priority score: volume * lift
    theme_summary["priority_score"] = (theme_summary["complaints_count"] * theme_summary["avg_sentiment_lift"]).round(3)
    return theme_summary, detail_df


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.title("AI Concierge — Rubric Agent (Baseline + Lift)")
st.caption(
    "This version computes a baseline sentiment per comment, asks the LLM for an impact category, "
    "maps that to a numeric lift, and derives predicted_post_sentiment = clip(baseline + lift)."
)

# Data upload section
st.header("1) Data")
uploaded_file = st.file_uploader("Upload CSV with a 'comment_text' column", type=["csv"])

data_df: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        data_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(data_df):,} rows from uploaded file.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        data_df = None
if data_df is None:
    st.info("No valid file uploaded. Using a small built‑in sample for demonstration.")
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
    st.dataframe(sample, use_container_width=True)
    data_df = sample.copy()

# Sidebar controls
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input(
    "OpenAI API key",
    type="password",
    placeholder="sk-...",
    key="api_key_input",
)
key_source = "secrets" if st.secrets.get("OPENAI_API_KEY") else ("input" if api_key_input else "missing")
st.sidebar.write("Key source:", key_source)
max_comments = st.sidebar.slider("Max comments to analyse", min_value=20, max_value=400, value=120, step=20)

# Analysis section
st.header("2) AI‑Driven Analysis (Rubric)")
exp = st.expander("How are the metrics calculated?")
with exp:
    st.markdown(
        """
        **Baseline sentiment**: The TextBlob polarity of the original comment (range -1 to 1).
        
        **Impact label**: The LLM chooses one of five categories reflecting how big a change the suggested fix will make: very_low, low, medium, high, very_high.
        
        **Lift (numeric)**: Each impact label maps to a numeric lift:
        - very_low → 0.05
        - low → 0.15
        - medium → 0.30
        - high → 0.50
        - very_high → 0.80
        The lift is optionally weighted by the model's confidence.
        
        **Predicted post‑sentiment**: `clip(baseline_sentiment + lift, -1, 1)`.
        
        **Complaints count**: How many comments were assigned to a theme.
        
        **Average sentiment lift**: The mean of per‑comment lifts within a theme.
        
        **Average predicted post‑sentiment**: The mean predicted sentiment after the fix.
        
        **Priority score**: `complaints_count × avg_sentiment_lift`. Bigger scores identify high‑volume, high‑impact themes.
        """
    )

if st.button("Run rubric analysis"):
    api_key = get_api_key()
    if not api_key:
        st.error("Please provide a valid OpenAI API key in the sidebar or via Streamlit secrets.")
    else:
        with st.spinner("Running rubric‑based analysis…"):
            summary_df, detail_df = analyse_with_rubric(data_df, api_key, max_comments=max_comments)
        if summary_df.empty:
            st.error("The analysis returned no results. Try a different dataset or increase the number of comments.")
        else:
            st.success("Analysis complete.")
            # Display theme summary
            st.subheader("Theme‑level summary")
            st.dataframe(summary_df, use_container_width=True)
            # Charts
            st.subheader("Visualisations")
            # Average lift per theme
            lift_chart = (
                alt.Chart(summary_df)
                .mark_bar()
                .encode(
                    y=alt.Y("theme:N", sort="-x", title="Theme"),
                    x=alt.X("avg_sentiment_lift:Q", title="Average sentiment lift"),
                    tooltip=[
                        "theme:N",
                        alt.Tooltip("avg_sentiment_lift:Q", format=".3f", title="Avg lift"),
                        "complaints_count:Q",
                        alt.Tooltip("avg_predicted_post_sentiment:Q", format=".3f", title="Avg post"),
                        "impact_mode:N",
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(lift_chart, use_container_width=True)

            # Priority score bar chart (top 10)
            top10 = summary_df.sort_values("priority_score", ascending=False).head(10)
            priority_chart = (
                alt.Chart(top10)
                .mark_bar()
                .encode(
                    y=alt.Y("theme:N", sort="-x", title="Theme"),
                    x=alt.X("priority_score:Q", title="Priority score"),
                    tooltip=[
                        "theme:N",
                        "complaints_count:Q",
                        alt.Tooltip("avg_sentiment_lift:Q", format=".3f", title="Avg lift"),
                        alt.Tooltip("priority_score:Q", format=".3f", title="Priority"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(priority_chart, use_container_width=True)

            # Scatter: volume vs avg lift, size by volume, colour by impact_mode
            scatter_chart = (
                alt.Chart(summary_df)
                .mark_circle()
                .encode(
                    x=alt.X("complaints_count:Q", title="Complaints (count)"),
                    y=alt.Y("avg_sentiment_lift:Q", title="Average lift"),
                    size=alt.Size("complaints_count:Q", legend=None, title="Volume"),
                    color=alt.Color("impact_mode:N", title="Dominant impact"),
                    tooltip=[
                        "theme:N",
                        "complaints_count:Q",
                        alt.Tooltip("avg_sentiment_lift:Q", format=".3f", title="Avg lift"),
                        alt.Tooltip("avg_predicted_post_sentiment:Q", format=".3f", title="Avg post"),
                        "impact_mode:N",
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(scatter_chart, use_container_width=True)

            # Histogram of per‑comment lifts
            if not detail_df.empty:
                st.subheader("Lift distribution (per comment)")
                hist_chart = (
                    alt.Chart(detail_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("sentiment_lift:Q", bin=alt.Bin(maxbins=30), title="Per‑comment lift"),
                        y=alt.Y("count()", title="Frequency"),
                        tooltip=[alt.Tooltip("count()", title="Count")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(hist_chart, use_container_width=True)

                # Baseline vs predicted post scatter
                st.subheader("Baseline vs predicted post sentiment")
                bp_chart = (
                    alt.Chart(detail_df)
                    .mark_circle(opacity=0.6)
                    .encode(
                        x=alt.X("baseline_sentiment:Q", title="Baseline sentiment"),
                        y=alt.Y("predicted_post_sentiment:Q", title="Predicted post sentiment"),
                        color=alt.Color("impact_label:N", title="Impact"),
                        tooltip=[
                            "theme:N",
                            "impact_label:N",
                            "confidence:Q",
                            "sentiment_lift:Q",
                            alt.Tooltip("predicted_post_sentiment:Q", format=".3f", title="Post"),
                            "suggestion:N",
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(bp_chart, use_container_width=True)

                # Sample detail rows
                st.subheader("Sample of analysed comments")
                st.dataframe(
                    detail_df[[
                        "theme",
                        "impact_label",
                        "confidence",
                        "baseline_sentiment",
                        "sentiment_lift",
                        "predicted_post_sentiment",
                        "suggestion",
                        "comment_text",
                    ]].head(50),
                    use_container_width=True
                )

# Footer
st.divider()
st.caption("Rubric Agent v1 • " + datetime.utcnow().strftime("%Y‑%m‑%d %H:%M UTC"))