import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import notification.notification as notification
from external_api import sentiment_rapid_spike  

load_dotenv()

# -----------------------------
# API CONFIG
# -----------------------------
RAPID_API_KEY = os.getenv("RAPID_API_KEY2")
HEADERS = {
    "x-rapidapi-key": RAPID_API_KEY,
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

SEARCH_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
REVIEW_URL = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

COUNTRY = "US"
SEARCH_PAGE = 1
REVIEW_PAGE = 1

# -----------------------------
# CATEGORY KEYWORDS
# -----------------------------
CATEGORY_KEYWORDS = {
    "Electricals_Power_Backup": ["inverter", "ups", "power backup", "generator"],
    "Home_Appliances": ["air conditioner", "refrigerator", "washing machine"],
    "Kitchen_Appliances": ["mixer", "grinder", "microwave"],
    "Computers_Tablets": ["laptop", "tablet"],
    "Mobile_Accessories": ["charger", "earphones", "power bank"],
    "Wearables": ["smartwatch", "fitness band"],
    "TV_Audio_Entertainment": ["smart tv", "speaker"],
}

# -----------------------------
# SENTIMENT MODEL (LOAD ONCE)
# -----------------------------
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Neutral"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_idx = torch.argmax(probs).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[sentiment_idx]

# -----------------------------
# SEARCH AMAZON PRODUCTS
# -----------------------------
def search_products(query):
    params = {
        "query": query,
        "page": SEARCH_PAGE,
        "country": COUNTRY,
        "sort_by": "RELEVANCE",
        "product_condition": "ALL",
        "is_prime": "false",
        "deals_and_discounts": "NONE"
    }

    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("products", [])

# -----------------------------
# FETCH REVIEWS BY ASIN
# -----------------------------
def fetch_reviews(asin):
    params = {
        "asin": asin,
        "country": COUNTRY,
        "page": REVIEW_PAGE,
        "sort_by": "TOP_REVIEWS",
        "star_rating": "ALL",
        "verified_purchases_only": "false",
        "images_or_videos_only": "false",
        "current_format_only": "false"
    }

    response = requests.get(REVIEW_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("reviews", [])

# -----------------------------
# MAIN FUNCTION (LIKE news.py)
# -----------------------------
def get_rapid_data():
    try:
        all_reviews = []

        for category, keywords in tqdm(CATEGORY_KEYWORDS.items(), desc="Categories"):
            for keyword in keywords:
                try:
                    products = search_products(keyword)

                    for product in products[:5]:
                        asin = product.get("asin")
                        if not asin:
                            continue

                        reviews = fetch_reviews(asin)

                        for r in reviews:
                            review_text = str(r.get("review_text", "")).strip()
                            review_title = str(r.get("review_title", "")).strip()

                            combined_text = (review_title + ". " + review_text).strip()

                            all_reviews.append({
                                "source": "Amazon_RapidAPI",
                                "category": category,
                                "keyword_used": keyword,
                                "asin": asin,
                                "product_title": product.get("title"),
                                "brand": product.get("brand"),
                                "price": product.get("price"),
                                "rating": r.get("rating"),
                                "review_title": review_title,
                                "review_text": review_text,
                                "review_date": r.get("review_date"),
                                "reviewer": r.get("reviewer_name"),
                                "verified_purchase": r.get("verified_purchase"),
                                "collected_at": datetime.utcnow(),
                                "combined_text": combined_text
                            })

                except Exception as e:
                    print(f"Error for keyword '{keyword}': {e}")

        df = pd.DataFrame(all_reviews)

        if df.empty:
            notification.send_mail("RapidAPI Review Alert", "No Amazon reviews collected today.")
            return

        # Remove duplicates
        df.drop_duplicates(subset=["asin", "review_text"], inplace=True)

        # Apply sentiment
        tqdm.pandas()
        df["sentiment_label"] = df["combined_text"].progress_apply(get_sentiment)
        df.drop(columns=["combined_text"], inplace=True)

        # Save output
        OUTPUT_FILE = "final data/rapid_reviews_with_sentiment.csv"

        if os.path.exists(OUTPUT_FILE):
            df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
        else:
            df.to_csv(OUTPUT_FILE, index=False)

        # Spike Detection
        alert_df = sentiment_rapid_spike.rapid_sentiment_spike(df)

        if alert_df.empty:
            notification.send_mail(
                "RapidAPI Review Alert",
                "Amazon reviews extracted successfully and no major weekly sentiment spikes detected."
            )
        else:
            notification.send_mail(
                "RapidAPI Review Alert",
                "Amazon reviews extracted successfully. Please find attached sentiment spike report.",
                alert_df
                )
            print("✅ Sentiment analysis completed and saved successfully.")

    except Exception as e:
        error_msg = (
        "❌ Failed to Extract News Data\n"
        f"Reason: {e}\n\n"
        f"{traceback.format_exc()}"
    )

    notification.send_mail("News Data Alert", error_msg)
    notification.send_slack_notification(error_msg)
        