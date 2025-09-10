# utils/speech_input.py

import streamlit as st

# Check if speech_recognition is available
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    st.warning("⚠️ Speech recognition not available. Install 'SpeechRecognition' and 'pyaudio' packages for voice input functionality.")

def get_voice_query():
    """
    Listens to microphone input and converts it to text using SpeechRecognition.
    Returns the recognized text or None if an error occurs.
    
    Note: This function requires 'SpeechRecognition' and 'pyaudio' packages.
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        st.error("Speech recognition is not available. Please install required packages:")
        st.code("pip install SpeechRecognition pyaudio", language="bash")
        return None
    
    try:
        r = sr.Recognizer()
        
        # Check if microphone is available
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                st.error("No microphone detected. Please connect a microphone and try again.")
                return None
        except:
            st.error("Unable to access microphone. Please check your system permissions.")
            return None
        
        with st.spinner("🎤 Listening... Speak now! (5 second timeout)"):
            try:
                with sr.Microphone() as source:
                    # Adjust for ambient noise
                    r.adjust_for_ambient_noise(source, duration=1)
                    
                    # Listen for speech with timeout
                    audio = r.listen(source, timeout=5, phrase_time_limit=10)

                # Convert speech to text using Google Web Speech API
                # Note: This requires an internet connection
                with st.spinner("🔄 Processing your speech..."):
                    text = r.recognize_google(audio)
                    return text.strip()
                    
            except sr.WaitTimeoutError:
                st.warning("⏱️ No speech detected within the timeout period. Please try again.")
                return None
            except sr.UnknownValueError:
                st.warning("🤷‍♂️ Could not understand the audio. Please speak clearly and try again.")
                return None
            except sr.RequestError as e:
                st.error(f"🌐 Speech recognition service error: {e}. Please check your internet connection.")
                return None
                
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during speech recognition: {e}")
        return None


def get_voice_query_with_alternatives():
    """
    Extended version that provides alternative implementations.
    This function can be expanded to include other speech recognition services.
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        st.info("💡 Alternative: You can type your query in the text area below instead of using voice input.")
        return None
    
    # Main speech recognition attempt
    result = get_voice_query()
    
    if result:
        return result
    
    # If main method fails, provide alternatives
    st.info("💡 Voice recognition tips:")
    st.write("• Ensure your microphone is working and properly connected")
    st.write("• Speak clearly and at a normal pace")
    st.write("• Try to minimize background noise")
    st.write("• Make sure you have a stable internet connection")
    st.write("• Consider typing your query instead")
    
    return None


def test_microphone():
    """
    Tests if the microphone is working and accessible.
    Returns True if microphone is available, False otherwise.
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        return False
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
        return True
    except:
        return False


# Example usage and demo queries
EXAMPLE_QUERIES = [
    "Show me the sales trend over time",
    "What is the total revenue by region",
    "Create a pie chart of product categories",
    "Display the relationship between price and quantity",
    "Show me a scatter plot of age versus income",
    "What are the top 5 customers by sales",
    "Create a bar chart of monthly expenses",
    "Show the distribution of ratings"
]

def get_example_queries():
    """
    Returns a list of example queries that users can try.
    """
    return EXAMPLE_QUERIES


def display_voice_help():
    """
    Displays helpful information about voice input functionality.
    """
    with st.expander("🎤 Voice Input Help", expanded=False):
        st.markdown("""
        ### How to Use Voice Input
        
        1. **Click the 🎤 Voice Input button**
        2. **Wait for the "Listening..." message**
        3. **Speak clearly** your data analysis request
        4. **Wait** for the system to process your speech
        
        ### Example Voice Queries
        You can say things like:
        """)
        
        for i, query in enumerate(EXAMPLE_QUERIES[:5], 1):
            st.write(f"{i}. *\"{query}\"*")
        
        st.markdown("""
        ### Troubleshooting
        - **No microphone detected**: Check if your microphone is connected and enabled
        - **Permission denied**: Allow microphone access in your browser settings  
        - **Can't understand audio**: Speak more clearly, reduce background noise
        - **Service error**: Check your internet connection (uses Google Speech API)
        - **Timeout**: The system waits 5 seconds for speech - try speaking immediately after clicking
        
        ### Technical Requirements
        - Internet connection (for speech processing)
        - Working microphone
        - Browser permissions for microphone access
        - Python packages: `SpeechRecognition`, `pyaudio`
        """)


# Alternative simple voice input for when full SR is not available
def get_manual_voice_simulation():
    """
    Provides a manual alternative when speech recognition is not available.
    Shows example queries that users can copy and paste.
    """
    st.info("🎤 Voice Input Not Available - Try These Example Queries Instead:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Chart Examples")
        if st.button("📈 Show sales trend", key="voice_sim_1"):
            return "Show me the sales trend over time"
        if st.button("🥧 Create pie chart", key="voice_sim_2"):
            return "Create a pie chart of categories"
        if st.button("📊 Bar chart", key="voice_sim_3"):
            return "Show total revenue by region"
        if st.button("🔍 Scatter plot", key="voice_sim_4"):
            return "Show relationship between price and quantity"
    
    with col2:
        st.subheader("📈 Analysis Examples")
        if st.button("🏆 Top performers", key="voice_sim_5"):
            return "What are the top 5 customers"
        if st.button("📅 Monthly data", key="voice_sim_6"):
            return "Show monthly trends"
        if st.button("📊 Distribution", key="voice_sim_7"):
            return "Show the distribution of values"
        if st.button("🔢 Summary stats", key="voice_sim_8"):
            return "Give me a summary of the data"
    
    return None


# # utils/speech_input.py

# import streamlit as st
# import speech_recognition as sr
# import requests
# import time
# import os
# import soundfile as sf
# import numpy as np

# # Check if required libraries are available
# try:
#     import speech_recognition as sr
#     import requests
#     import soundfile as sf
#     SPEECH_RECOGNITION_AVAILABLE = True
# except ImportError as e:
#     SPEECH_RECOGNITION_AVAILABLE = False
#     st.warning(f"⚠️ Voice input dependencies not available: {e}. Please install 'SpeechRecognition', 'pyaudio', 'requests', and 'soundfile'.")

# # Try to get the AssemblyAI API key securely from .streamlit/secrets.toml
# try:
#     # utils/speech_input.py
#     # utils/speech_input.py
#     ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
#     USE_ASSEMBLYAI = True
# except KeyError:
#     ASSEMBLYAI_API_KEY = None
#     USE_ASSEMBLYAI = False
#     st.info("💡 AssemblyAI API key not found. Using the default Google Web Speech API instead. Please add the key to your `.streamlit/secrets.toml` file to use AssemblyAI.")


# def get_voice_query():
#     """
#     Listens to microphone input and converts it to text.
#     Uses the AssemblyAI API if a key is provided, otherwise falls back to Google Web Speech API.
#     Returns the recognized text or None if an error occurs.
#     """
#     if not SPEECH_RECOGNITION_AVAILABLE:
#         st.error("Speech recognition is not available. Please install required packages:")
#         st.code("pip install SpeechRecognition pyaudio requests soundfile", language="bash")
#         return None
    
#     try:
#         r = sr.Recognizer()
        
#         try:
#             mic_list = sr.Microphone.list_microphone_names()
#             if not mic_list:
#                 st.error("No microphone detected. Please connect a microphone and try again.")
#                 return None
#         except:
#             st.error("Unable to access microphone. Please check your system permissions.")
#             return None
            
#         with st.spinner("🎤 Listening... Speak now! (5 second timeout)"):
#             try:
#                 with sr.Microphone() as source:
#                     r.adjust_for_ambient_noise(source, duration=1)
#                     audio = r.listen(source, timeout=5)
#             except sr.WaitTimeoutError:
#                 st.warning("⏱️ No speech detected within the timeout period. Please try again.")
#                 return None
#             except Exception as e:
#                 st.error(f"An error occurred during audio capture: {e}")
#                 return None
        
#         # Use AssemblyAI if key is present
#         if USE_ASSEMBLYAI:
#             with st.spinner("🔄 Sending audio to AssemblyAI for transcription..."):
#                 temp_audio_file = "temp_audio.wav"
#                 try:
#                     audio_data_np = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
#                     sf.write(temp_audio_file, audio_data_np, audio.sample_rate)
                    
#                     headers = {'authorization': ASSEMBLYAI_API_KEY, 'content-type': 'application/json'}
                    
#                     # Step 1: Upload the file
#                     with open(temp_audio_file, 'rb') as f:
#                         response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=f)
#                         upload_url = response.json().get('upload_url')
                    
#                     if not upload_url:
#                         st.error("Failed to upload audio to AssemblyAI.")
#                         return None
                    
#                     # Step 2: Submit for transcription
#                     json = {'audio_url': upload_url}
#                     response = requests.post('https://api.assemblyai.com/v2/transcript', json=json, headers=headers)
#                     transcript_id = response.json().get('id')
                    
#                     if not transcript_id:
#                         st.error("Failed to start transcription.")
#                         return None
                    
#                     # Step 3: Poll for the result
#                     polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
#                     while True:
#                         polling_response = requests.get(polling_endpoint, headers=headers)
#                         transcription_result = polling_response.json()
                        
#                         if transcription_result.get('status') == 'completed':
#                             return transcription_result['text']
#                         elif transcription_result.get('status') == 'failed':
#                             st.error(f"Transcription failed: {transcription_result.get('error')}")
#                             return None
#                         else:
#                             time.sleep(3)
                            
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Network error during AssemblyAI API call: {e}")
#                     return None
#                 finally:
#                     if os.path.exists(temp_audio_file):
#                         os.remove(temp_audio_file)
        
#         # Fallback to Google Web Speech API
#         else:
#             with st.spinner("🔄 Processing your speech with Google Web Speech API..."):
#                 text = r.recognize_google(audio)
#                 return text.strip()
            
#     except sr.UnknownValueError:
#         st.warning("🤷‍♂️ Could not understand the audio. Please speak clearly and try again.")
#         return None
#     except sr.RequestError as e:
#         st.error(f"🌐 Speech recognition service error: {e}. Please check your internet connection.")
#         return None
#     except Exception as e:
#         st.error(f"❌ An unexpected error occurred: {e}")
#         return None


# def get_voice_query_with_alternatives():
#     """
#     Extended version that provides alternative implementations.
#     This function can be expanded to include other speech recognition services.
#     """
#     if not SPEECH_RECOGNITION_AVAILABLE:
#         st.info("💡 Alternative: You can type your query in the text area below instead of using voice input.")
#         return None
    
#     # Main speech recognition attempt
#     result = get_voice_query()
    
#     if result:
#         return result
    
#     # If main method fails, provide alternatives
#     st.info("💡 Voice recognition tips:")
#     st.write("• Ensure your microphone is working and properly connected")
#     st.write("• Speak clearly and at a normal pace")
#     st.write("• Try to minimize background noise")
#     st.write("• Make sure you have a stable internet connection")
#     st.write("• Consider typing your query instead")
    
#     return None


# def test_microphone():
#     """
#     Tests if the microphone is working and accessible.
#     Returns True if microphone is available, False otherwise.
#     """
#     if not SPEECH_RECOGNITION_AVAILABLE:
#         return False
    
#     try:
#         r = sr.Recognizer()
#         with sr.Microphone() as source:
#             r.adjust_for_ambient_noise(source, duration=0.5)
#         return True
#     except:
#         return False


# # Example usage and demo queries
# EXAMPLE_QUERIES = [
#     "Show me the sales trend over time",
#     "What is the total revenue by region",
#     "Create a pie chart of product categories",
#     "Display the relationship between price and quantity",
#     "Show me a scatter plot of age versus income",
#     "What are the top 5 customers by sales",
#     "Create a bar chart of monthly expenses",
#     "Show the distribution of ratings"
# ]

# def get_example_queries():
#     """
#     Returns a list of example queries that users can try.
#     """
#     return EXAMPLE_QUERIES


# def display_voice_help():
#     """
#     Displays helpful information about voice input functionality.
#     """
#     with st.expander("🎤 Voice Input Help", expanded=False):
#         st.markdown("""
#         ### How to Use Voice Input
        
#         1. **Click the 🎤 Voice Input button**
#         2. **Wait for the "Listening..." message**
#         3. **Speak clearly** your data analysis request
#         4. **Wait** for the system to process your speech
        
#         ### Example Voice Queries
#         You can say things like:
#         """)
        
#         for i, query in enumerate(EXAMPLE_QUERIES[:5], 1):
#             st.write(f"{i}. *\"{query}\"*")
        
#         st.markdown("""
#         ### Troubleshooting
#         - **No microphone detected**: Check if your microphone is connected and enabled
#         - **Permission denied**: Allow microphone access in your browser settings  
#         - **Can't understand audio**: Speak more clearly, reduce background noise
#         - **Service error**: Check your internet connection (uses Google Speech API)
#         - **Timeout**: The system waits 5 seconds for speech - try speaking immediately after clicking
        
#         ### Technical Requirements
#         - Internet connection (for speech processing)
#         - Working microphone
#         - Browser permissions for microphone access
#         - Python packages: `SpeechRecognition`, `pyaudio`, `requests`, `soundfile`
#         """)


# # Alternative simple voice input for when full SR is not available
# def get_manual_voice_simulation():
#     """
#     Provides a manual alternative when speech recognition is not available.
#     Shows example queries that users can copy and paste.
#     """
#     st.info("🎤 Voice Input Not Available - Try These Example Queries Instead:")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📊 Chart Examples")
#         if st.button("📈 Show sales trend", key="voice_sim_1"):
#             return "Show me the sales trend over time"
#         if st.button("🥧 Create pie chart", key="voice_sim_2"):
#             return "Create a pie chart of categories"
#         if st.button("📊 Bar chart", key="voice_sim_3"):
#             return "Show total revenue by region"
#         if st.button("🔍 Scatter plot", key="voice_sim_4"):
#             return "Show relationship between price and quantity"
    
#     with col2:
#         st.subheader("📈 Analysis Examples")
#         if st.button("🏆 Top performers", key="voice_sim_5"):
#             return "What are the top 5 customers"
#         if st.button("📅 Monthly data", key="voice_sim_6"):
#             return "Show monthly trends"
#         if st.button("📊 Distribution", key="voice_sim_7"):
#             return "Show the distribution of values"
#         if st.button("🔢 Summary stats", key="voice_sim_8"):
#             return "Give me a summary of the data"
    
#     return None