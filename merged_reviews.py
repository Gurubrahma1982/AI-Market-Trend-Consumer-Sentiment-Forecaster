import pandas as pd
import re

# -----------------------------
# 1) LOAD FILES
# -----------------------------
df_lda = pd.read_csv("final data/category_wise_lda_output_with_topic_labels.csv")
df_rapid = pd.read_csv("final data/rapid_api_reviews_final.csv")


# -----------------------------
# 2) SENTIMENT FUNCTION (rating -> sentiment)
# -----------------------------
def sentiment_from_rating(r):
    try:
        r = float(r)
    except:
        return ""

    if r in [1, 2]:
        return "Negative"
    elif r == 3:
        return "Neutral"
    elif r in [4, 5]:
        return "Positive"
    return ""


# -----------------------------
# 3) DATE CONVERT FUNCTION (for both files -> dd-mm-yyyy)
# -----------------------------
def convert_to_dd_mm_yyyy(date_value):
    """
    Converts any date format into DD-MM-YYYY
    Examples:
    09/15/2022  -> 15-09-2022
    10-09-2022  -> 10-09-2022
    May 12, 2023 -> 12-05-2023
    """
    if pd.isna(date_value):
        return ""

    date_value = str(date_value).strip()

    # If date like "May 12, 2023"
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:,)?\s+(\d{4})\b", date_value)
    if m:
        date_str = m.group(0)
        try:
            dt = pd.to_datetime(date_str)
            return dt.strftime("%d-%m-%Y")
        except:
            return date_value

    # If date like 09/15/2022 or 10-09-2022
    try:
        dt = pd.to_datetime(date_value, errors="coerce", dayfirst=False)
        if pd.isna(dt):
            return date_value
        return dt.strftime("%d-%m-%Y")
    except:
        return date_value


# -----------------------------
# 4) FROM LDA OUTPUT FILE TAKE:
# source, review_text, review_date, sentiment, category
# -----------------------------
df_lda_final = pd.DataFrame({
    "source": df_lda["source"],
    "category": df_lda["category"],
    "review text": df_lda["review_text"],
    "sentiment": df_lda["sentiment_label"],
    "review date": df_lda["review_date"].apply(convert_to_dd_mm_yyyy),   # ✅ FIXED
})


# -----------------------------
# 5) FROM RAPID API FILE TAKE:
# source, label(category), review_text, rating->sentiment, review_date
# -----------------------------
df_rapid_final = pd.DataFrame({
    "source": df_rapid["source"],
    "category": df_rapid["label"],
    "review text": df_rapid["review_text"],
    "sentiment": df_rapid["rating"].apply(sentiment_from_rating),
    "review date": df_rapid["review_date"].apply(convert_to_dd_mm_yyyy),  # ✅ FIXED
})


# -----------------------------
# 6) COMBINE BOTH FILES
# -----------------------------
final_df = pd.concat([df_lda_final, df_rapid_final], ignore_index=True)


# -----------------------------
# 7) SAVE OUTPUT FILE
# -----------------------------
final_df.to_csv("combined_review.csv", index=False, encoding="utf-8")
print("Final dataset saved as combined_review.csv")
