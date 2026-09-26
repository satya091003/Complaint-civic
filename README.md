Municipal grievance portals in most cities rely on citizens manually selecting a category and staff manually triaging thousands of text complaints — a slow, inconsistent, and easy-to-misroute process. This project builds an end-to-end solution that makes both sides of that workflow faster and smarter.

On the citizen-facing side, users can report a civic issue (potholes, water supply problems, power outages, garbage, drainage, etc.) in whichever way is easiest for them — typing a description, uploading a voice note that gets automatically transcribed, and attaching a photo of the site, which is checked for basic quality (too dark or blurry) so staff receive usable evidence. A machine learning model (TF-IDF vectorization + Logistic Regression) trained on labeled complaint text suggests the most likely category in real time, and a keyword-based triage system flags high-priority issues (safety hazards, accidents, health risks) for faster attention — though the citizen always confirms the final category themselves.

On the administrative side, a live dashboard aggregates all submitted complaints into category breakdowns, priority distributions, a time-trend chart, and a geographic hotspot map, so city staff can quickly spot recurring problem areas and track each complaint's status from Open to Resolved.

Tech stack: Python, Streamlit, scikit-learn, pandas, Plotly, OpenCV, SpeechRecognition, SQLite.

Real-world relevance: This mirrors the exact pipeline used by systems like NYC's 311 service or India's Swachhata app — text/voice/photo intake, ML-assisted triage, and a public-facing analytics layer — making it directly extensible to a real municipal deployment with real complaint data.
