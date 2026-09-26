"""
app.py
------
Civic Complaint Dashboard — a Streamlit app where citizens can file a
complaint (category + photo + text/voice description) and city staff can
monitor trends and hotspots.

Run with:
    streamlit run app.py
"""

import datetime
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    AUDIO_DIR,
    IMAGE_DIR,
    check_image_quality,
    fetch_complaints,
    init_db,
    insert_complaint,
    save_uploaded_file,
    score_priority,
    transcribe_audio,
    update_status,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

st.set_page_config(page_title="Civic Complaint Dashboard", page_icon="🏙️", layout="wide")

CATEGORIES = [
    "Roads", "Water Supply", "Electricity", "Garbage", "Drainage",
    "Street Lighting", "Public Safety", "Parks & Public Spaces", "Other",
]

# Simulated ward/area coordinates so the hotspot map works without a
# geocoding API. Replace with your city's real ward coordinates.
AREA_COORDS = {
    "MG Road": (16.5062, 80.6480),
    "Gandhi Nagar": (16.5140, 80.6350),
    "Patel Nagar": (16.4980, 80.6600),
    "Lake View Colony": (16.5200, 80.6200),
    "Ring Road": (16.5300, 80.6550),
    "Old Town": (16.4900, 80.6300),
    "Riverside": (16.5100, 80.6700),
    "Industrial Estate": (16.5400, 80.6100),
    "Green Park": (16.5000, 80.6450),
    "Central Market": (16.5050, 80.6500),
    "Hill View": (16.5250, 80.6400),
    "Sector 12": (16.4950, 80.6550),
}

init_db()


@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


model = load_model()

st.title("🏙️ Civic Complaint Dashboard")
st.caption("Report a civic issue in seconds, or explore complaint trends across the city.")

tab_submit, tab_dashboard = st.tabs(["📝 File a Complaint", "📊 City Dashboard"])

# =======================================================================
# TAB 1 — File a complaint
# =======================================================================
with tab_submit:
    st.subheader("Report an issue")
    st.write("Fill in the details below. You can describe the problem by typing, "
             "by voice note, or both — whichever is easier for you.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        area = st.selectbox("Area / Locality", list(AREA_COORDS.keys()))

        description = st.text_area(
            "Describe the problem",
            placeholder="E.g. Streetlight near the bus stop has not worked for two weeks...",
            height=120,
        )

        st.markdown("**Or record a voice note describing the issue**")
        audio_file = st.file_uploader(
            "Upload a voice note (wav, mp3, m4a)", type=["wav", "mp3", "m4a", "ogg"]
        )
        transcribed_text = ""
        if audio_file is not None:
            st.audio(audio_file)
            if st.button("🎙️ Transcribe voice note"):
                with st.spinner("Transcribing..."):
                    audio_path = save_uploaded_file(audio_file, AUDIO_DIR)
                    transcribed_text = transcribe_audio(audio_path)
                st.session_state["transcribed_text"] = transcribed_text

        if "transcribed_text" in st.session_state:
            st.text_area(
                "Transcribed text (editable)",
                key="transcribed_text",
                height=80,
            )

        st.markdown("**Upload a photo of the problem area**")
        image_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            st.image(image_file, caption="Preview", use_container_width=True)

    with col_right:
        st.markdown("**Category**")
        suggested_category = None
        full_text_for_model = (description or "") + " " + st.session_state.get("transcribed_text", "")

        if model is not None and full_text_for_model.strip():
            if st.button("✨ Suggest category from my description"):
                suggested_category = model.predict([full_text_for_model])[0]
                st.session_state["suggested_category"] = suggested_category

        default_index = 0
        if "suggested_category" in st.session_state and st.session_state["suggested_category"] in CATEGORIES:
            st.success(f"Suggested category: **{st.session_state['suggested_category']}** "
                       f"(you can change this if it's wrong)")
            default_index = CATEGORIES.index(st.session_state["suggested_category"])

        category = st.selectbox("Confirm the complaint category", CATEGORIES, index=default_index)

        st.markdown("---")
        st.markdown("**Contact (optional)**")
        st.text_input("Name", key="citizen_name")
        st.text_input("Phone / Email", key="citizen_contact")

    st.markdown("---")
    submit = st.button("🚀 Submit Complaint", type="primary", use_container_width=True)

    if submit:
        final_text = (description or "").strip()
        transcribed = st.session_state.get("transcribed_text", "").strip()
        combined_text = (final_text + " " + transcribed).strip()

        if not combined_text:
            st.error("Please describe the problem using text or a voice note before submitting.")
        else:
            image_path = None
            if image_file is not None:
                image_path = save_uploaded_file(image_file, IMAGE_DIR)
                warning = check_image_quality(image_path)
                if warning:
                    st.warning(warning)

            audio_path = None
            if audio_file is not None:
                audio_path = save_uploaded_file(audio_file, AUDIO_DIR)

            predicted_category = None
            if model is not None:
                predicted_category = model.predict([combined_text])[0]

            priority = score_priority(combined_text)
            lat, lon = AREA_COORDS.get(area, (None, None))

            record = {
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "category": category,
                "predicted_category": predicted_category,
                "priority": priority,
                "area": area,
                "lat": lat,
                "lon": lon,
                "description": final_text,
                "transcribed_text": transcribed,
                "image_path": image_path,
                "audio_path": audio_path,
                "status": "Open",
            }
            insert_complaint(record)

            st.success(
                f"Complaint submitted! Category: **{category}** · "
                f"Priority: **{priority}**. Thank you for reporting this."
            )
            st.session_state.pop("transcribed_text", None)
            st.session_state.pop("suggested_category", None)

# =======================================================================
# TAB 2 — Dashboard
# =======================================================================
with tab_dashboard:
    st.subheader("City-wide complaint overview")

    records = fetch_complaints()
    if not records:
        st.info("No complaints submitted yet. File one in the first tab to see it here.")
    else:
        df = pd.DataFrame(records)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["date"] = df["created_at"].dt.date

        # --- Filters ---
        f1, f2, f3 = st.columns(3)
        with f1:
            cat_filter = st.multiselect("Filter by category", sorted(df["category"].unique()))
        with f2:
            area_filter = st.multiselect("Filter by area", sorted(df["area"].unique()))
        with f3:
            status_filter = st.multiselect("Filter by status", sorted(df["status"].unique()))

        filtered = df.copy()
        if cat_filter:
            filtered = filtered[filtered["category"].isin(cat_filter)]
        if area_filter:
            filtered = filtered[filtered["area"].isin(area_filter)]
        if status_filter:
            filtered = filtered[filtered["status"].isin(status_filter)]

        # --- KPI row ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Complaints", len(filtered))
        k2.metric("Open", int((filtered["status"] == "Open").sum()))
        k3.metric("High Priority", int((filtered["priority"] == "High").sum()))
        k4.metric("Resolved", int((filtered["status"] == "Resolved").sum()))

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            cat_counts = filtered["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig_cat = px.bar(
                cat_counts, x="category", y="count", title="Complaints by Category",
                color="category",
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with c2:
            fig_pri = px.pie(
                filtered, names="priority", title="Priority Breakdown",
                color="priority",
                color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
            )
            st.plotly_chart(fig_pri, use_container_width=True)

        trend = filtered.groupby("date").size().reset_index(name="count")
        fig_trend = px.line(trend, x="date", y="count", markers=True, title="Complaints Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("### 📍 Complaint Hotspot Map")
        map_df = filtered.dropna(subset=["lat", "lon"])
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", color="priority", hover_name="area",
                hover_data=["category", "status"],
                color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"},
                zoom=11, height=450,
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No location data to show on the map yet.")

        st.markdown("### 📋 Complaint Log")
        for _, row in filtered.iterrows():
            with st.expander(
                f"#{row['id']} · {row['category']} · {row['area']} · "
                f"{row['priority']} priority · {row['status']}"
            ):
                st.write(f"**Submitted:** {row['created_at']}")
                text_shown = row["description"] or row["transcribed_text"] or "(no description)"
                st.write(f"**Description:** {text_shown}")
                if row.get("predicted_category") and row["predicted_category"] != row["category"]:
                    st.caption(f"Model-suggested category was: {row['predicted_category']}")
                if row.get("image_path") and os.path.exists(row["image_path"]):
                    st.image(row["image_path"], width=300)
                if row.get("audio_path") and os.path.exists(row["audio_path"]):
                    st.audio(row["audio_path"])

                new_status = st.selectbox(
                    "Update status",
                    ["Open", "In Progress", "Resolved"],
                    index=["Open", "In Progress", "Resolved"].index(row["status"]),
                    key=f"status_{row['id']}",
                )
                if new_status != row["status"]:
                    if st.button("Save status", key=f"save_{row['id']}"):
                        update_status(row["id"], new_status)
                        st.rerun()
