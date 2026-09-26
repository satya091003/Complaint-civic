"""
utils.py
--------
Shared helper functions for the Civic Complaint Dashboard:
 - SQLite storage
 - Rule-based priority scoring
 - Voice note transcription (speech-to-text)
 - Basic image validation
"""

import os
import sqlite3
import datetime
from pathlib import Path

DB_PATH = "complaints.db"
IMAGE_DIR = Path("uploads/images")
AUDIO_DIR = Path("uploads/audio")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            category TEXT NOT NULL,
            predicted_category TEXT,
            priority TEXT NOT NULL,
            area TEXT NOT NULL,
            lat REAL,
            lon REAL,
            description TEXT NOT NULL,
            transcribed_text TEXT,
            image_path TEXT,
            audio_path TEXT,
            status TEXT NOT NULL DEFAULT 'Open'
        )
        """
    )
    conn.commit()
    conn.close()


def insert_complaint(record: dict):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO complaints
        (created_at, category, predicted_category, priority, area, lat, lon,
         description, transcribed_text, image_path, audio_path, status)
        VALUES (:created_at, :category, :predicted_category, :priority, :area,
                :lat, :lon, :description, :transcribed_text, :image_path,
                :audio_path, :status)
        """,
        record,
    )
    conn.commit()
    conn.close()


def fetch_complaints():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM complaints ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_status(complaint_id: int, status: str):
    conn = get_connection()
    conn.execute("UPDATE complaints SET status = ? WHERE id = ?", (status, complaint_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Priority scoring (simple, transparent, rule-based)
# ---------------------------------------------------------------------

HIGH_PRIORITY_KEYWORDS = [
    "fire", "electrocution", "accident", "danger", "dangerous", "urgent",
    "emergency", "collapse", "sewage overflow", "gas leak", "flood",
    "child", "children", "hospital", "sick", "injury", "injured",
]

MEDIUM_PRIORITY_KEYWORDS = [
    "leak", "broken", "damaged", "overflow", "unsafe", "hazard",
    "blocked", "no water", "power cut", "not working",
]


def score_priority(text: str) -> str:
    """Very simple keyword-based triage. Swap this out for a model trained
    on historical resolution-time data once you have real records."""
    t = text.lower()
    if any(word in t for word in HIGH_PRIORITY_KEYWORDS):
        return "High"
    if any(word in t for word in MEDIUM_PRIORITY_KEYWORDS):
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------
# Voice note transcription
# ---------------------------------------------------------------------

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes an uploaded voice note to text.

    Uses the `speech_recognition` library with Google's free Web Speech API
    (requires internet access on the machine running the app). Audio is
    first converted to WAV via pydub/ffmpeg since recognize_google requires
    WAV/FLAC/AIFF input.

    Returns an error message string (not an exception) on failure, so the
    calling UI code can display it directly without crashing the app.
    """
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
    except ImportError:
        return "[Transcription unavailable: install 'SpeechRecognition' and 'pydub' + ffmpeg]"

    try:
        wav_path = audio_file_path
        if not audio_file_path.lower().endswith(".wav"):
            sound = AudioSegment.from_file(audio_file_path)
            wav_path = str(Path(audio_file_path).with_suffix(".wav"))
            sound.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "[Could not understand audio clearly — please re-record or type instead]"
    except sr.RequestError:
        return "[Speech service unavailable — check your internet connection]"
    except Exception as e:
        return f"[Transcription failed: {e}]"


# ---------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------

def save_uploaded_file(uploaded_file, target_dir: Path) -> str:
    """Saves a Streamlit UploadedFile object to disk with a timestamped
    filename and returns the saved path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{uploaded_file.name}".replace(" ", "_")
    dest = target_dir / safe_name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def check_image_quality(image_path: str) -> str:
    """Flags very dark or very blurry images using OpenCV so citizens get
    immediate feedback to re-upload a clearer photo. Returns a warning
    string, or an empty string if the image looks fine."""
    try:
        import cv2
    except ImportError:
        return ""

    img = cv2.imread(image_path)
    if img is None:
        return "Could not read the image file — please try a different photo."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    warnings = []
    if brightness < 40:
        warnings.append("the photo looks very dark")
    if blur_score < 60:
        warnings.append("the photo looks blurry")

    if warnings:
        return "Heads up: " + " and ".join(warnings) + ". Consider re-uploading a clearer photo."
    return ""
