from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from collections import Counter
import joblib
import config

# Load trained model and vectorizer
model = joblib.load(r"saved_model\Restaurant_review_model.joblib")
vectorizer = joblib.load(r"saved_model\count_v_res.joblib")

# Food keywords
import pandas as pd 
food_csv = r"C:\Users\Hp\Desktop\Food_Digital_Menu\Recomendaiton.csv"
food_keywords = pd.read_csv(food_csv)['food_name'].tolist()
#please add the menu items from our Digital Menu(all of them in list format)


# Connect to MySQL
# create a config file and inside that add your own credentials of database
db = mysql.connector.connect(
    host=config.HOST_NAME,
    user=config.USER_NAME,
    password=config.PASSWORD,
    database=config.DB_NAME
)

print("✅ Connected to MySQL!")

cursor = db.cursor()

# FastAPI instance
app = FastAPI()

class Review(BaseModel):
    review_text: str

import re
from fastapi import HTTPException

@app.post("/add_review")
def add_review(review: Review):
    try:
        # --- Split into sub-sentences based on stop chars ---
        stop_list = r"[.,!;]"
        sub_sentences = re.split(stop_list, review.review_text)
        sub_sentences = [s.strip() for s in sub_sentences if s.strip()]  # clean empty ones

        sent_preds = []
        for sub in sub_sentences:
            X = vectorizer.transform([sub]).toarray()
            pred = model.predict(X)[0]  # 1=Positive, 0=Negative
            sent_preds.append(pred)

        # --- Apply sentiment logic ---
        if all(p == 1 for p in sent_preds):
            sentiment = "Positive"
        elif all(p == 0 for p in sent_preds):
            sentiment = "Negative"
        else:
            sentiment = "Mixed"

        # --- Insert review + final sentiment into MySQL ---
        sql = "INSERT INTO reviews (review_text, sentiment) VALUES (%s, %s)"
        cursor.execute(sql, (review.review_text, sentiment))
        db.commit()

        return {
            "message": "Review added successfully!",
            "predicted_sentiment": sentiment
        }

    except mysql.connector.Error as db_err:
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    


import re

@app.get("/recommend_food")
def recommend_food():
    try:
        # Fetch only unprocessed reviews
        cursor.execute("SELECT id, review_text FROM reviews WHERE processed = FALSE")
        reviews = cursor.fetchall()

        for r in reviews:
            review_id, review_text = r

            # --- Split review into sub-sentences ---
            stop_list = r'[.!;]'
            sub_sentences = [s.strip() for s in re.split(stop_list, review_text) if s.strip()]

            # --- Predict sentiment for each sub-sentence ---
            sent_preds = []
            for sub in sub_sentences:
                X_sub = vectorizer.transform([sub]).toarray()
                pred = model.predict(X_sub)[0]
                sent_preds.append(pred)

            # --- Determine final sentiment ---
            if all(p == 1 for p in sent_preds):
                final_p = 1  # Positive
            elif all(p == 0 for p in sent_preds):
                final_p = 0  # Negative
            else:
                final_p = 2  # Mixed

            # --- Update DB only for Positive or Negative ---
            if final_p in (0, 1):
                # Normalize: avoid duplicates in same review
                matched_keywords = set()
                for keyword in food_keywords:
                    if keyword.lower() in review_text.lower():
                        matched_keywords.add(keyword.lower())

                for keyword in matched_keywords:
                    cursor.execute("SELECT id, count FROM feature_food WHERE food = %s", (keyword,))
                    row = cursor.fetchone()

                    if row:  # Exists in fixed menu
                        db_id, db_count = row

                        if final_p == 1:  
                            new_count = db_count + 1
                        elif final_p == 0: 
                            new_count = db_count - 1
                            if new_count < 0:  
                                new_count = 0

                        cursor.execute(
                            "UPDATE feature_food SET count = %s WHERE id = %s",
                            (new_count, db_id)
                        )
                        db.commit()
                    # else: do nothing (skip keywords not in menu)

            # --- Mark review as processed ---
            cursor.execute("UPDATE reviews SET processed = TRUE WHERE id = %s", (review_id,))
            db.commit()

        # --- Fetch final top recommended foods from DB ---
        cursor.execute("SELECT food, count FROM feature_food ORDER BY count DESC")
        top_recommended = cursor.fetchall()

        return {"top_recommended_foods": top_recommended}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in recommendation: {str(e)}")

