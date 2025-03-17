"""Voice processing utilities for the Loan Chatbot."""

import os
import tempfile
import speech_recognition as sr

def process_voice_input(groq_client, audio_file_path=None, model="whisper-large-v3-turbo"):
    """
    Process voice input using Groq's transcription API.
    
    Args:
        groq_client: Initialized Groq client
        audio_file_path: Path to audio file, if None, captures from microphone
        model: Transcription model to use
        
    Returns:
        Transcribed text from the voice input
    """
    try:
        recognizer = sr.Recognizer()
        
        if audio_file_path:
            # Process existing audio file with Groq
            print(f"Processing audio file {audio_file_path}...")
            with open(audio_file_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(audio_file_path, file.read()),
                    model=model,
                    response_format="verbose_json",
                )
            transcribed_text = transcription.text
        else:
            # Use microphone and save to temp file
            print("Listening... Speak now.")
            temp_audio_path = os.path.join(tempfile.gettempdir(), "audio_input.wav")

            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source)

            # Save audio to temporary file
            with open(temp_audio_path, "wb") as f:
                f.write(audio.get_wav_data())

            # Process with Groq
            with open(temp_audio_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(temp_audio_path, file.read()),
                    model=model,
                    response_format="verbose_json",
                )

            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            transcribed_text = transcription.text

        return transcribed_text

    except Exception as e:
        print(f"Error processing voice input: {e}")
        return None