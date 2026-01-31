import requests
import pandas as pd
from datetime import datetime
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from notification import notification
import os
import traceback
from external_api import sentiment_reddit_spike
from external_api import reddit_api


tqdm.pandas()

# -----------------------------
# Reddit Search Map
# -----------------------------
reddit_search_map = {
    "Electricals_Power_Backup": "inverter OR UPS OR power backup",
    "Home_Appliances": "home appliances OR washing machine OR refrigerator",
    "Kitchen_Appliances": "kitchen appliances OR air fryer OR mixer grinder",
    "Furniture": "furniture OR sofa OR bed",
    "Home_Storage_Organization": "storage organizer OR wardrobe OR shelf",
    "Computers_Tablets": "laptop OR desktop OR tablet",
    "Mobile_Accessories": "mobile accessories OR phone case OR charger",
    "Wearables": "smartwatch OR fitness band OR wearable",
    "TV_Audio_Entertainment": "television OR smart TV OR speakers",
    "Networking_Devices": "router OR modem OR WiFi",
    "Toys_Kids": "kids toys OR baby toys OR educational toys",
    "Gardening_Outdoor": "gardening OR outdoor tools OR plants",
    "Kitchen_Dining": "kitchen dining OR cookware OR dinner set",
    "Mens_Clothing": "mens clothing OR men fashion OR shirts",
    "Footwear": "footwear OR shoes OR sneakers",
    "Beauty_Personal_Care": "beauty OR skincare OR personal care",
    "Security_Surveillance": "CCTV OR security camera OR surveillance",
    "Office_Printer_Supplies": "printer OR ink cartridge OR office supplies",
    "Software": "software OR apps OR technology",
    "Fashion_Accessories": "fashion accessories OR bags OR watches"
}

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def reddit_api():
    try:
        # -----------------------------
        # Reddit API Config
        # -----------------------------
        headers = {
            "User-Agent": "Mozilla/5.0 (TrendAnalysisBot)"
        }

        BASE_URL = "https://www.reddit.com/search.json"
        all_rows = []

        # -----------------------------
        # SCRAPE REDDIT DATA
        # -----------------------------
        for label, query in reddit_search_map.items():
            print(f"🔍 Searching Reddit for: {label}")

            params = {
                "q": query,
                "limit": 100
            }

            response = requests.get(BASE_URL, headers=headers, params=params)

            if response.status_code != 200:
                print(f"⚠️ Failed request for {label}")
                continue

            data = response.json()

            for post in data["data"]["children"]:
                post_data = post["data"]

                all_rows.append({
                    "source": "Reddit",
                    "category_label": label,
                    "search_query": query,
                    "title": post_data.get("title", ""),
                    "selftext": post_data.get("selftext", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "created_date": datetime.utcfromtimestamp(
                        post_data.get("created_utc", 0)
                    )
                })

            time.sleep(2)

        # -----------------------------
        # CREATE DATAFRAME
        # -----------------------------
        df = pd.DataFrame(all_rows)
        df = df[df["selftext"].str.strip() != ""].reset_index(drop=True)

        # -----------------------------
        # SENTIMENT ANALYSIS
        # -----------------------------
        print("🤖 Loading sentiment model...")

        MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        LABELS = ["negative", "neutral", "positive"]
        NUMERIC_MAP = {"negative": -1, "neutral": 0, "positive": 1}

        df["combined_text"] = (
            df["title"].fillna("") + " " + df["selftext"].fillna("")
        ).str.slice(0, 500)

        def get_sentiment(text):
            if not text.strip():
                return pd.Series(["neutral", 0.0, 0])

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

            label = LABELS[probs.argmax().item()]
            score = probs.max().item()
            numeric = NUMERIC_MAP[label]

            return pd.Series([label, score, numeric])

        print("📊 Running sentiment analysis...")
        df[["sentiment_label", "sentiment_score", "sentiment_numeric"]] = (
            df["combined_text"].progress_apply(get_sentiment)
        )

        # -----------------------------
        # SAVE DATA
        # -----------------------------
        os.makedirs("final data", exist_ok=True)
        df.to_excel("final data/reddit_category_trend_data.xlsx", index=False)
        print("✅ Saved reddit_category_trend_data.xlsx")

        # -----------------------------
        # SENTIMENT SPIKE DETECTION
        # -----------------------------
        result_df = sentiment_reddit_spike.reddit_sentiment_spike(df)

        if result_df.empty:
            notification.send_mail(
                subject="Reddit Data Extracted",
                text="Reddit data extracted successfully. No major weekly sentiment spikes detected."
            )
        else:
            notification.send_mail(
                subject="Reddit Sentiment Spike Detected",
                text="Please find the attached weekly Reddit sentiment spike & trend shift report.",
                df=result_df
            
            )

    except Exception as e:
        print("❌ Failed to save reddit data:", e)

        error_msg = (
            "❌ Reddit Data Extraction Failed\n"
            f"Reason: {e}\n\n"
            f"{traceback.format_exc()}"
        )

        # 📧 Email
        notification.send_mail(
            subject="Reddit Data Extraction Failed",
            text=error_msg
        )

        # 💬 Slack
        notification.send_slack_notification(error_msg)


# Uncomment to test directly
# if __name__ == "__main__":
#     reddit_api()
