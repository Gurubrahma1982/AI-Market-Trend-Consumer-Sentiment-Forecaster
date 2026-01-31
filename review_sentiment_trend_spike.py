import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = r"final data/category_wise_lda_output_with_topic_labels.csv"


DATE_COL = "review_date"
SENTIMENT_COL = "sentiment_label"
CATEGORY_COL = "category"


WEEK_WINDOW = 2               # rolling over 2 weeks
SPIKE_THRESHOLD = 0.3          # realistic spike threshold
TREND_SHIFT_THRESHOLD = 0.3    # realistic trend shift threshold


LABEL_MAP = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}


# -----------------------------
# LOAD & CLEAN DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)


df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL, SENTIMENT_COL, CATEGORY_COL])
df = df.sort_values(DATE_COL)


# Map sentiment to numeric
df["sentiment_score"] = df[SENTIMENT_COL].map(LABEL_MAP)


# -----------------------------
# WEEKLY AGGREGATION (CATEGORY-WISE)
# -----------------------------
weekly_cat = (
    df.groupby([CATEGORY_COL, df[DATE_COL].dt.to_period("W")])["sentiment_score"]
      .mean()
      .reset_index(name="weekly_sentiment")
)


weekly_cat[DATE_COL] = weekly_cat[DATE_COL].dt.start_time
weekly_cat = weekly_cat.sort_values([CATEGORY_COL, DATE_COL])


# -----------------------------
# ROLLING WINDOW CALCULATIONS
# -----------------------------
weekly_cat["rolling_avg"] = (
    weekly_cat
    .groupby(CATEGORY_COL)["weekly_sentiment"]
    .rolling(WEEK_WINDOW, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)


weekly_cat["prev_rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["rolling_avg"].shift(1)
weekly_cat["delta"] = weekly_cat["rolling_avg"] - weekly_cat["prev_rolling_avg"]
weekly_cat["prev_delta"] = weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)


# -----------------------------
# FILTER: LAST 2 WEEKS ONLY
# (after rolling calculations)
# -----------------------------
latest_week = weekly_cat[DATE_COL].max()
last_two_weeks_start = latest_week - pd.Timedelta(weeks=1)


weekly_cat_recent = weekly_cat[
    weekly_cat[DATE_COL] >= last_two_weeks_start
]


# -----------------------------
# ALERT DETECTION
# -----------------------------
alerts = []


for _, row in weekly_cat_recent.iterrows():
    if pd.isna(row["delta"]):
        continue


    # 🔴 Negative Spike
    if row["delta"] <= -SPIKE_THRESHOLD:
        alerts.append({
            "date": row[DATE_COL],
            "category": row[CATEGORY_COL],
            "alert_type": "NEGATIVE SPIKE",
            "change": round(row["delta"], 3)
        })


    # 🟢 Positive Spike
    if row["delta"] >= SPIKE_THRESHOLD:
        alerts.append({
            "date": row[DATE_COL],
            "category": row[CATEGORY_COL],
            "alert_type": "POSITIVE SPIKE",
            "change": round(row["delta"], 3)
        })


    # 🔄 Trend Shift (direction flip)
    if (
        pd.notna(row["prev_delta"]) and
        (row["delta"] * row["prev_delta"] < 0) and
        abs(row["delta"]) >= TREND_SHIFT_THRESHOLD
    ):
        alerts.append({
            "date": row[DATE_COL],
            "category": row[CATEGORY_COL],
            "alert_type": "TREND SHIFT",
            "change": round(row["delta"], 3)
        })


# -----------------------------
# OUTPUT
# -----------------------------
alert_df = pd.DataFrame(alerts)


if alert_df.empty:
    print("✅ No major weekly sentiment spikes or trend shifts detected in the last 2 weeks.")
else:
    print("🚨 WEEKLY SENTIMENT ALERTS (LAST 2 WEEKS)")
    print(alert_df.sort_values(["date", "category"]).reset_index(drop=True))



