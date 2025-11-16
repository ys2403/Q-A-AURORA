"""
eda_analysis.py
---------------
Exploratory Data Analysis (EDA) for AURORA_TEST message dataset.

This script analyzes the messages_pretty.json dataset to extract:
1. Basic message and user statistics
2. Encoding and text quality issues
3. Date mentions and formats
4. Sensitive data presence
5. User activity imbalance
6. Topic distribution based on keywords

Author: ystheaidev
Date: 2025-11-15
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd

# -----------------------------
# Load Data
# -----------------------------
with open("messages_pretty.json", "r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("items", data)
df = pd.DataFrame(items)

# -----------------------------
# Basic Overview
# -----------------------------
print("=== DATA OVERVIEW ===")
print(f"Total Messages: {len(df)}")
print(f"Unique Users: {df['user_id'].nunique()}")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Date Range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

msg_per_user = df.groupby("user_name")["message"].count()
print(f"\nAverage Messages per User: {msg_per_user.mean():.1f}")
print(msg_per_user.sort_values(ascending=False))

# -----------------------------
# Encoding Issue Detection
# -----------------------------
encoding_issues = df["message"].str.contains(r"[â€™â€]", regex=True)
encoding_rate = encoding_issues.mean() * 100
print(f"\nEncoding issues found in {encoding_rate:.2f}% of messages")
if encoding_rate > 0:
    print(df.loc[encoding_issues, "message"].head())

# -----------------------------
# Incomplete or Truncated Messages
# -----------------------------
incomplete = df["message"].str.strip().str.len() < 10
print(f"\nIncomplete/short messages: {incomplete.sum()} ({incomplete.mean()*100:.2f}%)")
print(df.loc[incomplete, ["user_name", "message"]].head())

# -----------------------------
# Sensitive Data Detection
# -----------------------------
phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
invoice_pattern = r"invoice|bill|charge|payment|card|account|number"
has_sensitive = df["message"].str.contains(phone_pattern, regex=True, case=False) | \
                df["message"].str.contains(invoice_pattern, regex=True, case=False)

print(f"\nSensitive data found in {has_sensitive.sum()} messages ({has_sensitive.mean()*100:.2f}%)")
print(df.loc[has_sensitive, ["user_name", "message"]].head())

# -----------------------------
# Date Mentions (absolute / relative)
# -----------------------------
absolute_dates = df["message"].str.contains(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", case=False)
relative_dates = df["message"].str.contains(r"\b(?:today|tomorrow|this|next|week|month|friday|saturday|sunday)\b", case=False)
print(f"\nMessages mentioning absolute dates: {absolute_dates.sum()} ({absolute_dates.mean()*100:.2f}%)")
print(f"Messages mentioning relative dates: {relative_dates.sum()} ({relative_dates.mean()*100:.2f}%)")

# -----------------------------
# Cultural Name Diversity (non-ASCII check)
# -----------------------------
non_ascii_names = df["user_name"].apply(lambda x: any(ord(c) > 127 for c in x))
print(f"\nUsers with non-ASCII names: {non_ascii_names.sum()} ({non_ascii_names.mean()*100:.2f}%)")
print(df.loc[non_ascii_names, "user_name"].unique())

# -----------------------------
# Topic Classification (rule-based)
# -----------------------------
topic_rules = {
    "Travel/Bookings": ["flight", "trip", "hotel", "tour", "book", "travel", "itinerary"],
    "Customer Service": ["thank", "issue", "problem", "help", "assist", "support"],
    "Account Updates": ["update", "profile", "account", "number", "contact", "change"],
    "Restaurant Reservations": ["dinner", "table", "restaurant", "reservation", "lunch"],
}

def classify_topic(msg: str) -> str:
    msg_lower = msg.lower()
    for topic, keywords in topic_rules.items():
        if any(k in msg_lower for k in keywords):
            return topic
    return "Other"

df["topic"] = df["message"].apply(classify_topic)
topic_dist = df["topic"].value_counts(normalize=True) * 100
print("\n=== TOPIC DISTRIBUTION ===")
print(topic_dist.round(2).to_string())

# -----------------------------
# Temporal Clustering Check
# -----------------------------
df["date"] = df["timestamp"].dt.date
msgs_per_day = df.groupby("user_name")["date"].nunique()
print(f"\nMedian active days per user: {msgs_per_day.median()}")

# -----------------------------
# Summary Table
# -----------------------------
summary = {
    "Total Messages": len(df),
    "Unique Users": df["user_id"].nunique(),
    "Date Range": f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}",
    "Avg Messages/User": round(msg_per_user.mean(), 1),
    "Encoding Issues (%)": round(encoding_rate, 2),
    "Sensitive Messages (%)": round(has_sensitive.mean()*100, 2),
    "Incomplete Messages (%)": round(incomplete.mean()*100, 2),
}

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# -----------------------------
# Optional: Export Clean Data
# -----------------------------
df.to_csv("messages_cleaned.csv", index=False)
print("\nCleaned dataset saved as messages_cleaned.csv")
