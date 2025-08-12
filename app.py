import streamlit as st
import pandas as pd
import random
from textblob import TextBlob

# Predefined sample comments for dummy data
SAMPLE_COMMENTS = [
    "Check-in took 25 minutes; line was too long.",
    "Staff were incredibly helpful and friendly.",
    "Room was dusty when I arrived.",
    "Loved the breakfast selection and service.",
    "WiFi kept disconnecting during meetings.",
]

def generate_dummy(n=200):
    comments = [random.choice(SAMPLE_COMMENTS) for _ in range(n)]
    is_complaint = [1 if ("too" in c.lower() or "dusty" in c.lower() or "disconnecting" in c.lower()) else 0 for c in comments]
    df = pd.DataFrame({
        "comment_text": comments,
        "is_complaint": is_complaint,
    })
    return df

def compute_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

st.title("AI Concierge Feedback Loop")
st.write("Upload a CSV of guest feedback or try the demo with synthetic data.")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
else:
    st.info("No file uploaded. Using demo data.")
    df = generate_dummy(200)

df["sentiment"] = df["comment_text"].apply(compute_sentiment)

complaint_counts = df["is_complaint"].value_counts().rename(index={0: "Praise", 1: "Complaint"})
st.bar_chart(complaint_counts)
st.dataframe(df.head())
