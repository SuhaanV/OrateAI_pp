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

aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Orate AI", page_icon="🎤", layout="wide")

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


@st.cache_resource
def load_pose_model():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose, mp_pose


@st.cache_resource
def load_posture_classifier():
    loaded = joblib.load('Final/model.pkl')
    if isinstance(loaded, dict):
        return loaded.get("model")
    return loaded


def send_google(transcript_text, audience, language_style, feedback_length):
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")
    prompt_text = f"Give professional public speaking feedback and tips to improve based on this speech. Also include an improved version of the speech. "
    if audience:
        prompt_text += f"Tailor the feedback and improved speech for a target audience of {audience}. "
    if language_style:
        prompt_text += f"Use a {language_style.lower()} language style appropriate for the audience.\n\n"
    if feedback_length:
        prompt_text += f"Use a {feedback_length.lower()} limit to give feedback.\n\n"
    prompt_text += transcript_text
    response = model.generate_content(prompt_text)
    return response.text


def pdf_export(transcript, sentiment_results, filler_data, ai_feedback):
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"orate_ai_report_{int(time.time())}.pdf")

    pdf = FPDF()
    pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
    pdf.add_page()

    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, "Orate AI - Speech Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Transcribed Text:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 8, transcript or "(no transcript)")
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Sentiment Analysis:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 12)
    if sentiment_results:
        for result in sentiment_results:
            line = f'"{result.text}" → {result.sentiment} ({result.confidence * 100:.1f}%)'
            pdf.multi_cell(0, 8, line)
    else:
        pdf.multi_cell(0, 8, "No sentiment data available.")
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Filler Words:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 12)
    if filler_data:
        for f in filler_data:
            line = f"'{f['text']}' at {f['time']}s"
            pdf.multi_cell(0, 8, line)
    else:
        pdf.multi_cell(0, 8, "No filler words detected.")
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "AI Feedback:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 11)
    pdf.multi_cell(0, 8, ai_feedback or "(no AI feedback)")

    pdf.output(filename)
    return filename


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

    with st.spinner("Generating PDF report..."):
        pdf_path = pdf_export(transcript.text, transcript.sentiment_analysis, filler_list, ai_feedback)
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
            input_data = pd.DataFrame([landmark_list])
            prediction = classifier.predict(input_data)
            posture_status = str(prediction[0])

        cv2.putText(image, str(posture_status), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Pose Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image, posture_status


st.title("🎤 Orate AI")
st.markdown("### Professional Public Speaking & Posture Analysis")

with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("final/assets/logo.png", width=200)
    else:
        st.markdown("## 🎤 Orate AI")

    st.markdown("---")
    st.subheader("⚙️ Settings")

    audience = st.text_input("Target Audience", placeholder="e.g., Business professionals")
    language_style = st.selectbox("Language Style", ["Understandable", "Scientific"])
    feedback_length = st.selectbox("Feedback Length", ["Summarised(200 words)", "Detailled(1000 words)"])

    st.markdown("---")
    st.subheader("💡 Quick Tips")

    tip_choice = st.radio("Select tip:", ["Posture", "Tone", "Confidence"], label_visibility="collapsed")

    if tip_choice == "Posture":
        st.info("Stand tall, keep your shoulders relaxed, and face your audience confidently.")
    elif tip_choice == "Tone":
        st.info("Vary your pitch and pace to keep your speech engaging.")
    else:
        st.info("Practice deep breathing before you speak to calm nerves.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎙️ Record Your Speech")

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="3x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("🚀 Analyze Speech", type="primary", use_container_width=True):
            pdf_path = process_audio(audio_bytes, audience, language_style, feedback_length)
            st.success("✅ Analysis complete!")

with col2:
    st.subheader("📹 Posture Monitor")

    pose, mp_pose = load_pose_model()
    mp_drawing = mp.solutions.drawing_utils
    classifier = load_posture_classifier()

    camera_input = st.camera_input("Live Posture Detection")

    if camera_input is not None:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        processed_frame, posture_status = process_posture_frame(
            frame, pose, mp_pose, mp_drawing, classifier
        )

        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st.image(processed_frame, use_container_width=True)

        if posture_status == "Good":
            st.success(f"✅ {posture_status} Posture")
        elif posture_status == "Bad":
            st.error(f"⚠️ {posture_status} Posture")
        else:
            st.info(f"📊 {posture_status}")

if st.session_state.transcript_text:
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Transcript", "😊 Sentiment", "🗣️ Filler Words", "💬 AI Feedback"])

    with tab1:
        st.text_area("Transcribed Text", st.session_state.transcript_text, height=300)

    with tab2:
        if st.session_state.sentiment_results:
            for result in st.session_state.sentiment_results:
                sentiment_icon = "😊" if result.sentiment == "POSITIVE" else "😐" if result.sentiment == "NEUTRAL" else "😟"
                st.markdown(
                    f"{sentiment_icon} **\"{result.text}\"** → {result.sentiment} ({result.confidence * 100:.1f}%)")
        else:
            st.info("No sentiment data available")

    with tab3:
        if st.session_state.filler_list:
            for word in st.session_state.filler_list:
                st.markdown(f"• **'{word['text']}'** at {word['time']}s")
        else:
            st.success("🎉 No filler words detected! Great job!")

    with tab4:
        st.markdown(st.session_state.ai_feedback)

    st.markdown("---")
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf",
                use_container_width=True
            )