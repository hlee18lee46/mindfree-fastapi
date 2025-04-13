from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import numpy as np
import cv2
from deepface import DeepFace
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict
from datetime import datetime
from pymongo import MongoClient, DESCENDING, ASCENDING

from openai import OpenAI
import google.generativeai as genai
import os
import json

from hashlib import sha256
import requests




# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# CORS: Allow all origins for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI")  # Put this in your .env
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["mindfree"]
collection = db["emotions"]
user_collection = db["users"]


# Pydantic model
class EmotionData(BaseModel):
    user_email: str
    emotions: Dict[str, float]
    timestamp: str  # ISO format expected

class EmotionRequest(BaseModel):
    image_data: str  # base64-encoded image string from webcam

class ChatRequest(BaseModel):
    user_input: str
    user_email: str

# 🔹 Chat endpoint using OpenAI GPT-3.5
# Endpoint

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Fetch latest 5 emotion logs
        emotion_logs = list(
            collection.find({"user_email": req.user_email})
            .sort("timestamp", DESCENDING)
            .limit(5)
        )

        # Format emotion log summaries
        emotion_summary = "\n".join(
            f"{i+1}. {log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} — " +
            ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in log['emotions'].items())
            for i, log in enumerate(emotion_logs)
        ) or "No emotion data available."

        # Add emotion summaries to system prompt
        system_prompt = f"""You are an empathetic and supportive AI therapist.
Here are the user's 5 most recent emotion summaries based on real-time facial expressions:

{emotion_summary}

Use this context to better understand and respond to the user's feelings."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.user_input}
            ]
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

from bson import ObjectId

@app.post("/log_emotion")
async def save_emotion(data: EmotionData):
    data_dict = data.dict()
    data_dict["timestamp"] = datetime.fromisoformat(data.timestamp)

    # Insert into MongoDB
    collection.insert_one(data_dict)

    # 🔍 Find userId from email
    user = user_collection.find_one({"email": data.user_email})
    if not user:
        return {"error": "User not found"}
    user_id_str = str(user["_id"])

    # 🔐 Generate SHA256 hash
    hash_input = json.dumps({
        "user_email": data.user_email,
        "timestamp": data.timestamp,
        "emotions": data.emotions
    }, sort_keys=True)
    hash_hex = sha256(hash_input.encode("utf-8")).hexdigest()

    # 🔗 Send hash + userId to Node.js Midnight anchor endpoint
    try:
        requests.post(
            "http://localhost:6300/api/midnight/anchor",
            json={"userId": user_id_str, "hash": hash_hex}
        )
    except Exception as e:
        print(f"❗ Failed to anchor to Midnight: {e}")

    return {"status": "saved", "anchored_hash": hash_hex}

@app.post("/log_emotion2")
async def save_emotion(data: EmotionData):
    data_dict = data.dict()
    data_dict["timestamp"] = datetime.fromisoformat(data.timestamp)  # Ensure it's a datetime object
    collection.insert_one(data_dict)
    return {"status": "saved"}

class MeditateRequest(BaseModel):
    user_email: str
    user_input: str


@app.post("/gemini/meditate")
async def meditate_recommendation(req: MeditateRequest = Body(...)):
    user_email = req.user_email
    user_input = req.user_input

    # Get latest emotion log
    log = collection.find_one(
        {"user_email": user_email},
        sort=[("timestamp", DESCENDING)]
    )

    if not log or "emotions" not in log:
        return {
            "title": "Ambient Meditation",
            "reason": "No recent emotion data found. Playing default calming track."
        }

    emotions = log["emotions"]
    emotion_context = ", ".join(f"{k}: {round(v * 100, 1)}%" for k, v in emotions.items())

    prompt = f"""
You are a meditation assistant. The user's recent emotion scores are:
{emotion_context}

Choose the best track from this list:
- Ambient Meditation (calm, floating)
- Clear Sky (hopeful, peaceful)
- In Meditation (relaxing, centered)
- Rainstick Cascade (soothing, peaceful, sad)
- Tibetan Bowl (deep, serene, focus)

Respond ONLY with a JSON object like:
{{"title": "...", "reason": "..."}}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(prompt)

    try:
        # Remove wrapping markdown (e.g. ```json\n...\n```)
        cleaned_text = result.text.strip().strip("`").replace("json", "").strip()
        parsed = json.loads(cleaned_text)
        return parsed
    except Exception as e:
        return {
            "title": "Ambient Meditation",
            "reason": f"Gemini failed to parse JSON. Error: {str(e)}. Raw: {result.text}"
        }
    
@app.get("/emotion/history")
def get_emotion_history(user_email: str = Query(...)):
    logs = list(collection.find({"user_email": user_email}).sort("timestamp", DESCENDING))
    for log in logs:
        log["_id"] = str(log["_id"])  # Convert ObjectId to string
    return logs

@app.get("/emotion/history_asc")
def get_emotion_history(user_email: str = Query(...)):
    logs = list(collection.find({"user_email": user_email}).sort("timestamp", ASCENDING))
    for log in logs:
        log["_id"] = str(log["_id"])  # Convert ObjectId to string
    return logs