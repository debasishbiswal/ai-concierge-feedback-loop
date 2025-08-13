"""
Streamlit application for analysing guest feedback.

This app replicates parts of the AI Concierge Feedback Loop demo.  It allows the user to upload
their own guest feedback CSV or generate a synthetic dataset if no file is provided.  A simple
sentiment score is computed using TextBlob and a few summary statistics are displayed.  The
synthetic data generation mirrors the schema used in the original notebook.
"""

import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import os
from textblob import TextBlob
# Import Altair for interactive charts
import altair as alt

# New imports for LangChain/OpenAI integration
try:
    # The LangChain modules are optional. If not installed the app will still work
    # Import ChatOpenAI from langchain_openai; import ChatPromptTemplate from langchain_core.prompts.
    # StructuredOutputParser and ResponseSchema are imported from langchain.output_parsers.
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema  # type: ignore
except ImportError:
    # Stub assignments so that static analysis won't fail if LangChain is absent
    ChatOpenAI = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    StructuredOutputParser = None  # type: ignore
    ResponseSchema = None  # type: ignore


# -----------------------------------------------------------------------------
# Configuration and data pools (copied from the original notebook)
# -----------------------------------------------------------------------------

PROPERTIES: List[tuple] = [
    ("Marriott Tampa Waterside", "Tampa", "FL", "Resort"),
    ("JW Marriott Orlando", "Orlando", "FL", "Convention"),
    ("Courtyard Miami Beach", "Miami", "FL", "Urban"),
    ("Ritz-Carlton Sarasota", "Sarasota", "FL", "Resort"),
    ("AC Hotel Tampa Downtown", "Tampa", "FL", "Urban"),
    ("Moxy Miami South Beach", "Miami", "FL", "Lifestyle"),
    ("Marriott Marquis Houston", "Houston", "TX", "Convention"),
    ("Westin Bonaventure Los Angeles", "Los Angeles", "CA", "Urban"),
    ("W New York Times Square", "New York", "NY", "Urban"),
    ("Sheraton Boston", "Boston", "MA", "Convention"),
]

CHANNELS: List[str] = ["in-stay chat", "post-stay email", "Bonvoy App", "TripAdvisor", "SMS", "Front desk survey"]

PURPOSES: List[str] = ["business", "leisure", "family", "conference"]

ROOM_TYPES: List[str] = ["standard", "deluxe", "suite"]

LOYALTY: List[str] = ["Non-member", "Member", "Silver", "Gold", "Platinum", "Titanium", "Ambassador"]

COMPLAINT_TEMPLATES: List[str] = [
    "Check-in took {delay} minutes; line was too long.",
    "Requested {item} twice and still waiting.",
    "Gym opens too {time} for my schedule.",
    "Breakfast line was {adj}; waited {delay} minutes.",
    "WiFi kept {issue} during meetings.",
    "Room was {adj2} when I arrived.",
    "AC was {issue2}; room never cooled properly.",
    "Pool hours end too {time} for families.",
    "Parking fee felt {adj} given the stay.",
    "Housekeeping missed {item} restock.",
]

VAR: Dict[str, List[str]] = {
    "delay": ["15", "25", "35", "45", "55", "65", "75"],
    "item": ["extra towels", "toiletries", "blanket", "water", "pillows"],
    "time": ["late", "early"],
    "adj": ["insane", "long", "excessive", "unexpected", "ridiculous"],
    "adj2": ["messy", "dusty", "unclean", "musty"],
    "issue": ["dropping", "disconnecting", "slow"],
    "issue2": ["noisy", "weak", "broken"],
}

PRAISE_TEMPLATES: List[str] = [
    "Staff were incredibly helpful and friendly.",
    "Room was spotless and the bed was comfortable.",
    "Great location and smooth check-in.",
    "Concierge solved my problem in minutes.",
    "Loved the breakfast selection and service.",
    "Lobby vibe and bar were fantastic.",
]

NON_EN: List[tuple] = [
    ("es", "El registro fue rápido y el personal muy amable."),
    ("es", "La fila del desayuno fue larga."),
    ("fr", "La chambre était propre et confortable."),
    ("de", "Check-in hat zu lange gedauert."),
    ("hi", "स्टाफ़ बहुत मददगार था।"),
]


# -----------------------------------------------------------------------------
# Helper functions for synthetic data generation
# -----------------------------------------------------------------------------

def fill_template(tpl: str) -> str:
    """Fill a complaint template with random variables."""
    out = tpl
    for k, vals in VAR.items():
        tok = "{" + k + "}"
        if tok in out:
            out = out.replace(tok, random.choice(vals))
    return out


def add_typos(text: str, p: float = 0.04) -> str:
    """Randomly add a repeated character to simulate a typo."""
    if random.random() > p or len(text) < 6:
        return text
    i = random.randrange(1, len(text) - 1)
    return text[:i] + text[i] + text[i] + text[i + 1 :]


def pick_date_2025() -> str:
    """Return a random date string in 2025 with weighted month probabilities."""
    month = random.choices(
        range(1, 13), weights=[6, 6, 7, 7, 8, 10, 11, 10, 8, 7, 6, 6]
    )[0]
    day = random.randint(1, 28)
    return f"2025-{month:02d}-{day:02d}"


def generate_synthetic_data(n_rows: int = 1000, seed: int = 123) -> pd.DataFrame:
    """Generate a synthetic guest feedback dataset of `n_rows` rows.

    The schema matches that used in the original notebook.  A smaller default
    number of rows is used here for quick local exploration.
    """
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for i in range(1, n_rows + 1):
        prop, city, state, ptype = random.choice(PROPERTIES)
        channel = random.choice(CHANNELS)
        purpose = random.choices(PURPOSES, weights=[0.45, 0.35, 0.15, 0.05])[0]
        room = random.choices(ROOM_TYPES, weights=[0.6, 0.3, 0.1])[0]
        tier = random.choices(LOYALTY, weights=[0.25, 0.25, 0.2, 0.15, 0.1, 0.04, 0.01])[0]
        date = pick_date_2025()

        # 70% complaint, 30% praise
        if random.random() < 0.7:
            text = fill_template(random.choice(COMPLAINT_TEMPLATES))
            is_complaint = 1
        else:
            text = random.choice(PRAISE_TEMPLATES)
            is_complaint = 0

        # ~3% multilingual feedback
        if random.random() < 0.03:
            _, text = random.choice(NON_EN)

        # Random typos for realism
        text = add_typos(text, 0.04)

        # Extract wait time if present
        wait = next((int(tok) for tok in text.split() if tok.isdigit()), 0)

        stay_len = random.choices(
            [1, 2, 3, 4, 5, 6, 7], weights=[0.25, 0.25, 0.2, 0.15, 0.08, 0.05, 0.02]
        )[0]
        adults = random.choice([1, 1, 2, 2, 2, 3])
        kids = random.choices([0, 0, 0, 1, 2], weights=[0.6, 0.25, 0.1, 0.03, 0.02])[0]

        rows.append(
            {
                "feedback_id": i,
                "property": prop,
                "city": city,
                "state": state,
                "property_type": ptype,
                "channel": channel,
                "created_at": date,
                "comment_text": text,
                "stay_purpose": purpose,
                "room_type": room,
                "loyalty_tier": tier,
                "stay_length_nights": stay_len,
                "party_adults": adults,
                "party_kids": kids,
                "observed_wait_minutes": wait,
                "is_complaint": is_complaint,
            }
        )

    df = pd.DataFrame(rows)
    return df


def compute_sentiment(text: str) -> float:
    """Return the polarity score of a text using TextBlob."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def analyse_feedback(df: pd.DataFrame, api_key: str) -> Dict[str, any]:
    """
    Analyse a dataframe of guest feedback using a language model via LangChain.

    This function performs three tasks on the first 100 comments (to keep costs in check):

    - Identify the top themes across the feedback.
    - Generate an actionable suggestion for each theme.
    - Provide an overall summary of sentiment and key takeaways.

    A dictionary with keys ``themes``, ``suggestions``, and ``overall_summary`` is returned.  If
    LangChain or the OpenAI modules are not available, the function raises a runtime error.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback data containing a ``comment_text`` column.
    api_key : str
        The OpenAI API key used by the ``ChatOpenAI`` model.

    Returns
    -------
    dict
        A dictionary with lists of themes and suggestions, and a summary string.
    """
    # Import lazily to avoid import-time failures when langchain is not installed
    if ChatOpenAI is None or ChatPromptTemplate is None or StructuredOutputParser is None:
        raise RuntimeError("LangChain and langchain-openai are required for analysis. Please install them in your environment.")
    if not isinstance(df, pd.DataFrame) or "comment_text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'comment_text' column.")
    # Take the first 100 comments to control API usage
    comments = df["comment_text"].astype(str).head(100).tolist()
    # Compose a single string with newline separated comments
    comment_block = "\n".join(f"- {c}" for c in comments)
    # Define the expected output schema
    schema = [
        ResponseSchema(name="themes", description="A list of the top themes identified in the feedback."),
        ResponseSchema(name="suggestions", description="Actionable suggestions corresponding to each theme."),
        ResponseSchema(name="overall_summary", description="A concise overall summary of the feedback sentiments and key takeaways."),
    ]
    parser = StructuredOutputParser.from_response_schemas(schema)
    format_instructions = parser.get_format_instructions()
    # Build a prompt template; instruct the model clearly and request JSON output according to the schema
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant tasked with analysing hotel guest feedback.\n"
            "Your job is to identify common themes, propose actionable suggestions to improve the guest experience,"
            " and summarise the overall sentiment. Use the user's comments exactly as provided."
        ),
        (
            "user",
            "Here is a list of guest comments:\n{comments}\n\n{format_instructions}"
        ),
    ])
    llm = ChatOpenAI(api_key=api_key, temperature=0)
    chain = prompt | llm | parser
    # Run the chain to get the structured output
    response = chain.invoke({"comments": comment_block, "format_instructions": format_instructions})
    # Ensure expected keys exist; fallback gracefully if not
    result = {
        "themes": response.get("themes", []),
        "suggestions": response.get("suggestions", []),
        "overall_summary": response.get("overall_summary", "")
    }
    return result


# -----------------------------------------------------------------------------
# Streamlit user interface
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AI Concierge Feedback Loop", page_icon="🛎️", layout="wide")
    st.title("AI Concierge Feedback Loop")
    st.write(
        "Upload the latest guest feedback data as a CSV file. If you don't have a file, "
        "the app will generate a dummy dataset that mimics the structure of the original notebook."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df):,} rows from your upload.")
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            st.stop()
    else:
        st.info("No file uploaded. Generating a synthetic data set for demo purposes.")
        df = generate_synthetic_data(1000)
        st.success(f"Generated {len(df):,} rows of dummy data.")

    # Compute sentiment if not already present
    if "sentiment" not in df.columns:
        st.write("Computing sentiment scores…")
        df["sentiment"] = df["comment_text"].astype(str).apply(compute_sentiment)

    # Summary statistics
        st.header("Summary Statistics")
        # Determine property column name if available
        property_col = "property" if "property" in df.columns else None
        # Display quick metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", int(len(df)))
        col2.metric("Unique Properties", int(df[property_col].nunique()) if property_col else 0)
        col3.metric("Avg sentiment", round(float(df["sentiment"].mean()), 3))
    
        # --- Interactive charts (Altair) ---
        st.header("Interactive Charts")
        # 1) Complaints vs Praises (robust to missing columns and strict typing)
        if "is_complaint" in df.columns:
            tmp_labels = df["is_complaint"].astype(int).map({1: "Complaints", 0: "Praises"}).fillna("Unknown")
        else:
            # Fallback: derive complaint label from sentiment being negative
            tmp_labels = (df["sentiment"] < 0).map({True: "Complaints", False: "Praises"})
        counts_df = (
            pd.DataFrame({"label": tmp_labels})
            .value_counts()
            .reset_index(name="count")
            .astype({"label": "string", "count": "int64"})
        )
        chart_counts = (
            alt.Chart(counts_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", sort=["Complaints", "Praises"], title=None),
                y=alt.Y("count:Q", title="Count"),
                tooltip=[alt.Tooltip("label:N", title="Type"), alt.Tooltip("count:Q", title="Count")],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(chart_counts, use_container_width=True)
    
        # 2) Average sentiment by property (if available)
        if property_col:
            prop_avg = (
                df.groupby(property_col, dropna=False)["sentiment"]
                .mean()
                .reset_index()
                .rename(columns={"sentiment": "avg_sentiment"})
                .astype({property_col: "string", "avg_sentiment": "float64"})
            )
            chart_prop = (
                alt.Chart(prop_avg)
                .mark_bar()
                .encode(
                    y=alt.Y(f"{property_col}:N", sort='-x', title="Property"),
                    x=alt.X("avg_sentiment:Q", title="Avg sentiment"),
                    tooltip=[
                        alt.Tooltip(f"{property_col}:N", title="Property"),
                        alt.Tooltip("avg_sentiment:Q", title="Avg sentiment", format=".3f"),
                    ],
                )
                .properties(height=420)
                .interactive()
            )
            st.altair_chart(chart_prop, use_container_width=True)
    
        # 3) Comments by channel (if channel column exists)
        if "channel" in df.columns:
            by_channel = (
                df["channel"].astype(str)
                .value_counts()
                .rename_axis("channel")
                .reset_index(name="count")
                .astype({"channel": "string", "count": "int64"})
            )
            chart_channel = (
                alt.Chart(by_channel)
                .mark_bar()
                .encode(
                    x=alt.X("channel:N", sort="-y", title="Channel"),
                    y=alt.Y("count:Q", title="Comments"),
                    tooltip=[alt.Tooltip("channel:N"), alt.Tooltip("count:Q")],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart_channel, use_container_width=True)
    
    st.header("Sample Data")
    st.dataframe(df.head(20))

    # Offer language-model analysis if LangChain and OpenAI are available
    st.header("AI‑Driven Analysis (optional)")
    st.write(
        "Enhance your exploration by using a large language model to identify themes, generate actionable suggestions,\n"
        "and summarise the feedback. Provide your OpenAI API key to run the analysis on the first 100 comments."
    )
    # Retrieve API key from Streamlit secrets if available, otherwise prompt user
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Enter your OpenAI API Key to enable AI‑powered analysis:",
            type="password",
            help="If left blank, the app will use local TextBlob analysis instead."
        )
    # If an API key is provided, set it as an environment variable for libraries that auto-discover
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Display a status badge showing where the key came from and package versions
    status_parts: List[str] = []
    # Determine the source of the API key
    if st.secrets.get("OPENAI_API_KEY", ""):
        status_parts.append("Key source: secrets")
    elif api_key:
        status_parts.append("Key source: input")
    else:
        status_parts.append("Key source: none")
    # Try to report installed package versions
    try:
        import langchain  # type: ignore
        import langchain_openai  # type: ignore
        import openai  # type: ignore
        lc_ver = getattr(langchain, "__version__", "unknown")
        lco_ver = getattr(langchain_openai, "__version__", "unknown")
        oa_ver = getattr(openai, "__version__", "unknown")
        status_parts.append(f"Versions — langchain {lc_ver}, langchain-openai {lco_ver}, openai {oa_ver}")
    except Exception:
        status_parts.append("Versions unavailable")
    # Show the status as a caption to keep it unobtrusive
    st.caption(" | ".join(status_parts))

    if st.button("Run Language Model Analysis"):
        if not api_key:
            st.error("Please enter a valid OpenAI API key before running the analysis.")
        else:
            try:
                with st.spinner("Running analysis with LangChain and OpenAI…"):
                    result = analyse_feedback(df, api_key)
                st.success("Analysis complete.")
                st.subheader("Top Themes")
                # Display as bullet points for readability
                for i, theme in enumerate(result.get("themes", []), 1):
                    st.markdown(f"{i}. {theme}")
                st.subheader("Actionable Suggestions")
                for i, sugg in enumerate(result.get("suggestions", []), 1):
                    st.markdown(f"{i}. {sugg}")
                st.subheader("Overall Summary")
                st.write(result.get("overall_summary", "No summary returned."))
            except Exception as exc:
                st.error(f"An error occurred during analysis: {exc}")

    # Allow user to download dummy data if generated
    if uploaded_file is None:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Dummy Data", csv, file_name="guest_feedback_dummy.csv")


if __name__ == "__main__":
    main()