from speechbrain.inference import SpeakerRecognition
import torchaudio
import numpy as np
from pydub import AudioSegment
from speechbrain.utils.fetching import fetch
import os

# Set torchaudio backend
torchaudio.set_audio_backend("soundfile")

# Load SpeechBrain's speaker recognition model with fetch_strategy set to "copy"
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"fetch_strategy": "copy"}  # Forces copying instead of symlink
)


# Convert any audio file to WAV (Mono, 16kHz)
def convert_to_wav(input_file: str, output_file: str = "converted_audio.wav", target_sample_rate: int = 16000) -> str:
    try:
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            return None

        # Load the audio file
        audio = AudioSegment.from_file(input_file)

        # Set to mono and desired sample rate
        audio = audio.set_channels(1).set_frame_rate(target_sample_rate)

        # Export as WAV
        audio.export(output_file, format="wav")

        # Debugging information
        print(f"Converted file saved at: {output_file}")
        print(f"Converted file exists: {os.path.exists(output_file)}")

        return output_file
    except Exception as e:
        print(f"Error converting file {input_file} to WAV: {e}")
        return None
# Extract speaker embeddings
def extract_embedding(file_path: str):
    try:
        # Ensure the input file is WAV format
        wav, sr = torchaudio.load(file_path)
        
        # Convert to correct sample rate if needed
        if sr != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
        
        embedding = verification_model.encode_batch(wav).squeeze().detach().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error extracting embedding from {file_path}: {e}")
        return None
