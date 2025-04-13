# 🧠 MindFree.AI – FastAPI Backend

This is the Python-based backend server for **MindFree.AI**, an intelligent, privacy-first AI therapy assistant. It powers real-time emotion detection, AI chatbot support, and Gemini-based meditation recommendations.

---

## 🚀 Features

- 📷 **Emotion Detection** from webcam input
- 🧠 **OpenAI GPT-3.5** for empathetic AI conversation
- 🎵 **Gemini-powered** meditation recommendations
- 📊 **Emotion data logging** with timestamping
- 🔐 **SHA-256 hashing** of data for blockchain anchoring
- 🪙 Communicates with Node.js server to anchor data to the **Midnight Blockchain**
- 🌐 CORS support for frontend integration

---

## 🧩 Tech Stack

| Tech       | Purpose                                     |
|------------|---------------------------------------------|
| **FastAPI**| Main web server (Python)                    |
| **MongoDB**| Stores user, emotion, and metadata records  |
| **OpenAI** | Chatbot functionality (GPT-3.5)             |
| **Gemini** | Meditation suggestions                      |
| **SHA-256**| Hashing emotion payloads before anchoring   |
| **requests**| Sends data to Node.js (Midnight anchor)    |

---

## 🛠️ Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/mindfree-fastapi.git
cd mindfree-fastapi

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Create .env file
MONGO_URI=mongodb+srv://<your-cluster>
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=your-gemini-api-key

### 4. Run the server
uvicorn main:app --reload --port 8000

### 🔗 Integration
This backend communicates with:
	•	Node.js server at http://localhost:6300/api/midnight/anchor to anchor hashes to the blockchain.
	•	MongoDB Atlas to persist user and emotion data.

## 🔍 Python Packages Used

| Package               | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `fastapi`             | Web framework for building APIs quickly with async support              |
| `uvicorn`             | ASGI server used to run the FastAPI application                         |
| `pymongo`             | MongoDB driver for Python – used to store and retrieve user/emotion data|
| `python-dotenv`       | Loads environment variables from `.env`                                 |
| `openai`              | Connects to OpenAI API (e.g., GPT-3.5) for AI-powered responses          |
| `google-generativeai`| Connects to Google's Gemini API for meditation suggestions              |
| `requests`            | Sends HTTP requests to Node.js server for Midnight anchoring            |
| `pydantic`            | Data validation and serialization via Python models                     |
| `numpy`, `opencv-python` *(optional)*| For image decoding and preprocessing (base64 → frame)    |


