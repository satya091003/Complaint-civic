# Civic Complaint Classifier & Dashboard

A Streamlit app where citizens report civic issues (potholes, water supply,
garbage, streetlights, etc.) using **text, a voice note, and a photo**, get
an **auto-suggested category** and **priority level**, and city staff can
monitor everything on a **live analytics dashboard** with a hotspot map.

## Features

- 📝 Citizen-friendly complaint form: category dropdown, area selector,
  free-text description, voice-note upload with transcription, photo upload
  with a basic quality check (flags dark/blurry photos)
- 🤖 ML-based category suggestion (TF-IDF + Logistic Regression) — citizen
  still confirms/overrides the final category
- ⚠️ Rule-based priority triage (High / Medium / Low) from keywords in the
  description
- 📊 Dashboard: category breakdown, priority split, complaints-over-time
  trend, and a hotspot map, with filters by category/area/status
- 🗂️ Status tracking (Open → In Progress → Resolved) stored in SQLite

## Project structure

```
civic-dashboard/
├── app.py                 # Streamlit app (UI + dashboard)
├── utils.py                # DB, priority scoring, transcription, image checks
├── train_model.py          # Trains the category classifier
├── generate_data.py        # Builds the synthetic training dataset
├── data/sample_complaints.csv
├── requirements.txt
└── uploads/                # Saved photos & voice notes (created at runtime)
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Voice transcription only) install ffmpeg, required by pydub
#    macOS:   brew install ffmpeg
#    Ubuntu:  sudo apt install ffmpeg
#    Windows: https://ffmpeg.org/download.html

# 4. Train the category classifier (creates model.pkl)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Important: replace the synthetic data with real complaints

`data/sample_complaints.csv` is **template-generated** so the classifier has
something to learn from out of the box — that's why `train_model.py` reports
100% accuracy (the templates are too clean/separable to be realistic). For
a real project submission:

1. Replace it with real historical complaints if you can get them (e.g.
   [NYC 311 Service Requests](https://opendata.cityofnewyork.us/) is public
   and well-documented, or check your local city's open-data portal).
2. Re-run `python train_model.py` and report the *real* precision/recall —
   including where the model struggles. That honesty is what makes a
   project look mature in a report or interview, not a perfect score.
3. Update `AREA_COORDS` in `app.py` with your actual city's ward/locality
   coordinates instead of the placeholder values.

## Notes on the voice transcription

`utils.transcribe_audio()` uses Google's free Web Speech API via the
`SpeechRecognition` package, which needs an internet connection at runtime.
For an offline-capable version, swap it for a local
[Whisper](https://github.com/openai/whisper) model — the function signature
(audio path in, text out) stays identical, so `app.py` doesn't need to change.

## Ideas to extend this into a stronger final project

- Swap the TF-IDF+LogisticRegression baseline for a fine-tuned DistilBERT
  and compare accuracy/latency in your report
- Train the priority score on real complaint **resolution times** instead of
  keyword rules
- Add authentication so citizens can track their own complaint's status
- Deploy publicly via [Streamlit Community Cloud](https://streamlit.io/cloud)
  (free) and put the live link in your resume/portfolio
- Add SHAP explanations for the category model so you can show *why* a
  complaint was classified a certain way
