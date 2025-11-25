import streamlit as st
import os
import time
import assemblyai as aai
import google.generativeai as genai
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from audio_recorder_streamlit import audio_recorder
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import toml

# Load configuration from secrets
try:
    config = st.secrets
    ASSEMBLY_API_KEY = config["api_keys"]["assemblyai"]
    GOOGLE_API_KEY = config["api_keys"]["google"]
    APP_USERNAME = config["auth"]["username"]
    APP_PASSWORD = config["auth"]["password"]
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Configure APIs
aai.settings.api_key = ASSEMBLY_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


# Authentication check
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Orate AI - Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if username == APP_USERNAME and password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.stop()


# Run authentication
check_authentication()

# Page configuration
st.set_page_config(page_title="Orate AI", page_icon="🎤", layout="wide")

# Session state initialization
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = []
if 'filler_list' not in st.session_state:
    st.session_state.filler_list = []
if 'ai_feedback' not in st.session_state:
    st.session_state.ai_feedback = ""
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'pose_model_loaded' not in st.session_state:
    st.session_state.pose_model_loaded = False
if 'posture_classifier' not in st.session_state:
    st.session_state.posture_classifier = None
if 'wpm' not in st.session_state:
    st.session_state.wpm = 0


@st.cache_resource
def load_pose_model():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose, mp_pose


@st.cache_resource
def load_posture_classifier():
    try:
        loaded = joblib.load('models/model.pkl')
        if isinstance(loaded, dict):
            model = loaded.get("model", None)
        else:
            model = loaded
        return model if model else None
    except Exception as e:
        st.warning(f"Posture classifier not available: {e}")
        return None


def calculate_wpm(transcript, audio_duration_ms):
    """Calculate words per minute from transcript

    Args:
        transcript: The transcribed text
        audio_duration_ms: Audio duration in milliseconds

    Returns:
        Words per minute (rounded to nearest integer)
    """
    if not transcript or audio_duration_ms <= 0:
        return 0

    word_count = len(transcript.split())
    duration_seconds = audio_duration_ms / 1000  # Convert ms to seconds
    duration_minutes = duration_seconds / 60  # Convert seconds to minutes

    if duration_minutes <= 0:
        return 0

    wpm = round(word_count / duration_minutes)
    return wpm


def send_google(transcript_text, audience, language_style, feedback_length):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    prompt_text = f"Give professional public speaking feedback and tips to improve based on this speech. Also include an improved version of the speech. Give the speech in human tone. "
    if audience:
        prompt_text += f"Tailor the feedback and improved speech for a target audience of {audience}. "
    if language_style:
        prompt_text += f"Use a {language_style.lower()} language style appropriate for the audience.\n\n"
    if feedback_length:
        prompt_text += f"Use a {feedback_length.lower()} limit to give feedback.\n\n"
    prompt_text += transcript_text
    response = model.generate_content(prompt_text)
    return response.text


def clean_text_for_pdf(text):
    """Clean text to be safe for PDF rendering"""
    if not text:
        return ""

    # Convert to string and handle special characters
    text = str(text)

    # Replace problematic characters
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--',  # Em dash
        '\u2026': '...',  # Ellipsis
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove any remaining non-ASCII characters that might cause issues
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def safe_multi_cell(pdf, text, line_height=8):
    """Safely write multi-line text to PDF with error handling"""
    try:
        if not text:
            pdf.multi_cell(0, line_height, "(no content)")
            return

        # Clean the text
        clean_text = clean_text_for_pdf(text)

        # Split into smaller chunks if too long
        max_chunk_length = 5000
        if len(clean_text) > max_chunk_length:
            chunks = [clean_text[i:i + max_chunk_length] for i in range(0, len(clean_text), max_chunk_length)]
            for chunk in chunks:
                pdf.multi_cell(0, line_height, chunk)
        else:
            pdf.multi_cell(0, line_height, clean_text)
    except Exception as e:
        pdf.multi_cell(0, line_height, f"(Content could not be rendered: {str(e)})")


def pdf_export(transcript, sentiment_results, filler_data, ai_feedback, wpm):
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"speech_report_{int(time.time())}.pdf")

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # Use standard Arial font
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Speech Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        # WPM Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Speaking Rate:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", '', 11)
        safe_multi_cell(pdf, f"Words Per Minute (WPM): {wpm}")

        # Provide context for WPM
        if wpm < 130:
            wpm_feedback = "Your speaking pace is slow. Consider speaking slightly faster."
        elif wpm > 170:
            wpm_feedback = "Your speaking pace is fast. Consider slowing down for clarity."
        else:
            wpm_feedback = "Your speaking pace is good (ideal range: 130-170 WPM)."
        safe_multi_cell(pdf, wpm_feedback)
        pdf.ln(5)

        # Transcript Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Transcribed Text:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", '', 11)
        safe_multi_cell(pdf, transcript or "(no transcript)")
        pdf.ln(5)

        # Sentiment Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Sentiment Analysis:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", '', 11)
        if sentiment_results:
            for result in sentiment_results:
                try:
                    text = clean_text_for_pdf(result.text)
                    sentiment = result.sentiment
                    confidence = result.confidence
                    line = f'"{text}" -> {sentiment} ({confidence * 100:.1f}%)'
                    safe_multi_cell(pdf, line)
                except Exception as e:
                    safe_multi_cell(pdf, f"(Sentiment data could not be rendered)")
        else:
            safe_multi_cell(pdf, "No sentiment data available.")
        pdf.ln(5)

        # Filler Words Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Filler Words:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", '', 11)
        if filler_data:
            for f in filler_data:
                try:
                    line = f"'{clean_text_for_pdf(f['text'])}' at {f['time']}s"
                    safe_multi_cell(pdf, line)
                except Exception as e:
                    safe_multi_cell(pdf, f"(Filler word data could not be rendered)")
        else:
            safe_multi_cell(pdf, "No filler words detected.")
        pdf.ln(5)

        # AI Feedback Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "AI Feedback:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", '', 11)
        safe_multi_cell(pdf, ai_feedback or "(no AI feedback)")

        pdf.output(filename)
        return filename

    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None


def process_audio(audio_bytes, audience, language_style, feedback_length):
    audio_file = "temp_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)

    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        sentiment_analysis=True,
        word_boost=["um", "uh", "like", "you know"],
        boost_param="high"
    )

    with st.spinner("Transcribing audio..."):
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_file)

    if transcript.status == "error":
        st.error(f"Transcription error: {transcript.error}")
        return None

    # Calculate WPM - audio_duration is in milliseconds
    audio_duration_ms = transcript.audio_duration
    wpm = calculate_wpm(transcript.text, audio_duration_ms)
    st.session_state.wpm = wpm

    # Extract filler words
    filler_list = []
    for word in transcript.words:
        if word.text.lower() in ["um", "uh", "like", "you know"]:
            time_sec = round(word.start / 1000, 2)
            filler_list.append({"text": word.text, "time": time_sec})

    with st.spinner("Getting AI feedback..."):
        ai_feedback = send_google(transcript.text, audience, language_style, feedback_length)

    st.session_state.transcript_text = transcript.text
    st.session_state.sentiment_results = transcript.sentiment_analysis
    st.session_state.filler_list = filler_list
    st.session_state.ai_feedback = ai_feedback

    pdf_path = None
    with st.spinner("Generating PDF report..."):
        pdf_path = pdf_export(transcript.text, transcript.sentiment_analysis, filler_list, ai_feedback, wpm)
        if pdf_path:
            st.session_state.pdf_path = pdf_path

    if os.path.exists(audio_file):
        os.remove(audio_file)

    return pdf_path


def process_posture_frame(frame, pose, mp_pose, mp_drawing, classifier):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    posture_status = "No Pose Detected"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_landmarks.landmark
        landmark_list = []
        for lm in landmarks:
            landmark_list.extend([lm.x, lm.y, lm.z])

        if classifier:
            try:
                input_data = pd.DataFrame([landmark_list])
                prediction = classifier.predict(input_data)
                posture_status = str(prediction[0])
            except Exception as e:
                posture_status = f"Prediction error: {str(e)}"
        else:
            posture_status = "Classifier not available"

        cv2.putText(image, str(posture_status), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Pose Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image, posture_status


# Main UI
st.title("Orate AI")

with st.sidebar:
    if os.path.exists("assets/sass.png"):
        st.image("assets/sass.png", width=300)

    st.markdown("---")
    st.subheader("Settings")

    audience = st.text_input("Target Audience", placeholder="e.g., Business professionals")
    language_style = st.selectbox("Language Style", ["Understandable", "Scientific"])
    feedback_length = st.selectbox("Feedback Length", ["Summarised(200 words)", "Detailled(1000 words)"])

    st.markdown("---")
    st.subheader("Quick Tips")

    tip_choice = st.radio("Select tip:", ["Posture", "Tone", "Confidence"], label_visibility="collapsed")

    if tip_choice == "Posture":
        st.info("Stand tall, keep your shoulders relaxed, and face your audience confidently.")
    elif tip_choice == "Tone":
        st.info("Vary your pitch and pace to keep your speech engaging.")
    else:
        st.info("Practice deep breathing before you speak to calm nerves.")

    # Logout button
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Load models
pose, mp_pose = load_pose_model()
mp_drawing = mp.solutions.drawing_utils
classifier = load_posture_classifier()

col_audio, col_video = st.columns([1, 1])

with col_audio:
    st.subheader("Audio & Analysis Controls")

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="3x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Analyze Speech", type="primary", use_container_width=True):
            try:
                pdf_path = process_audio(audio_bytes, audience, language_style, feedback_length)
                st.success("Analysis complete! Results loaded below.")
                if not pdf_path:
                    st.warning("PDF generation failed, but your results are displayed below.")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.info("Your results may still be visible below if transcription completed.")

    st.markdown("---")

with col_video:
    st.subheader("Real-time Posture Detection")

    run_camera = st.checkbox("Start Camera Feed", key="start_camera_main")

    stframe = st.empty()
    status_text = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Check your camera permissions.")
        else:
            stop_button = st.button("Stop Camera Feed", key="stop_camera_main", use_container_width=True)

            while run_camera and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame")
                    break

                processed_frame, posture_status = process_posture_frame(
                    frame, pose, mp_pose, mp_drawing, classifier
                )

                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame, channels="RGB", use_container_width=True)
                status_text.markdown(f"**Current Posture:** **{posture_status}**")

                time.sleep(0.03)

            cap.release()
            st.info("Camera feed stopped")
    else:
        status_text.info("Check 'Start Camera Feed' to begin posture detection.")

# Display results
if st.session_state.transcript_text:
    st.markdown("---")
    st.subheader("Analysis Results")

    # Display WPM prominently
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words Per Minute (WPM)", st.session_state.wpm)
    with col2:
        st.metric("Word Count", len(st.session_state.transcript_text.split()))
    with col3:
        st.metric("Filler Words", len(st.session_state.filler_list))

    tab_transcript, tab_sentiment, tab_filler, tab_feedback = st.tabs(
        ["Transcript", "Sentiment", "Filler Words", "AI Feedback"]
    )

    with tab_transcript:
        st.text_area("Transcribed Text", st.session_state.transcript_text, height=300)

    with tab_sentiment:
        if st.session_state.sentiment_results:
            for result in st.session_state.sentiment_results:
                st.markdown(f"**\"{result.text}\"** -> **{result.sentiment}** ({result.confidence * 100:.1f}%)")
        else:
            st.info("No sentiment data available")

    with tab_filler:
        if st.session_state.filler_list:
            for word in st.session_state.filler_list:
                st.markdown(f"**'{word['text']}'** at **{word['time']}s**")
        else:
            st.success("No filler words detected! Great job!")

    with tab_feedback:
        st.markdown(st.session_state.ai_feedback)

    st.markdown("---")

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf",
                use_container_width=True
            )
    elif st.session_state.transcript_text:
        st.info("PDF generation encountered an issue, but you can copy the results above.")