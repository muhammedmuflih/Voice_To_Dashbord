from gtts import gTTS
import os
import streamlit as st

def speak_text(text: str):
    """
    Converts text to speech and plays it.
    Uses gTTS (Google Text-to-Speech) and Streamlit's audio component.
    """
    if not text:
        st.warning("No text provided for speech.")
        return

    try:
        tts = gTTS(text=text, lang='en')
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)

        st.audio(audio_file, format="audio/mp3", start_time=0)
        os.remove(audio_file)  # Clean up temporary file
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}. Please ensure you have an active internet connection and the gTTS library installed.")