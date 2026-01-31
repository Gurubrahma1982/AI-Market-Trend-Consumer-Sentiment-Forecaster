import pandas as pd

def rapid_sentiment_spike(df):
    DATE_COL = "review_date"
    SENTIMENT_COL = "sentiment_label"
    CATEGORY_COL = "category"

    WEEK_WINDOW = 2
    SPIKE_THRESHOLD = 0.20
    TREND_SHIFT_THRESHOLD = 0.20

    LABEL_MAP = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL)

    df["sentiment_score"] = df[SENTIMENT_COL].map(LABEL_MAP)

    weekly_cat = (
        df.groupby([CATEGORY_COL, df[DATE_COL].dt.to_period("W")])["sentiment_score"]
        .mean()
        .reset_index(name="weekly_sentiment")
    )

    weekly_cat[DATE_COL] = weekly_cat[DATE_COL].dt.start_time

    weekly_cat["rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["weekly_sentiment"] \
        .rolling(WEEK_WINDOW, min_periods=1).mean() \
        .reset_index(level=0, drop=True)

    weekly_cat["prev_rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)["rolling_avg"].shift(1)
    weekly_cat["delta"] = weekly_cat["rolling_avg"] - weekly_cat["prev_rolling_avg"]
    weekly_cat["prev_delta"] = weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)

    alerts = []

    for _, row in weekly_cat.iterrows():
        if pd.isna(row["delta"]):
            continue

        if row["delta"] <= -SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "NEGATIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        if row["delta"] >= SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "POSITIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        if pd.notna(row["prev_delta"]) and (row["delta"] * row["prev_delta"] < 0) and \
           (abs(row["delta"]) >= TREND_SHIFT_THRESHOLD):
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "TREND SHIFT",
                "change": round(row["delta"], 3)
            })

    alert_df = pd.DataFrame(alerts)

    if alert_df.empty:
        print("✅ No major weekly sentiment spikes or trend shifts detected.")
    else:
        print("🚨 MAJOR WEEKLY SENTIMENT ALERTS")
        print(alert_df.sort_values(["date", "category"]).reset_index(drop=True))

    return alert_df
