import librosa
import numpy as np

# Feature extraction function for audio files (MFCC)
def extract_features_audio(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)  # Load audio file
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  
        mfccs_mean = np.mean(mfccs, axis=1)  # Calculate the mean of the MFCCs
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None  # Return None in case of an error
