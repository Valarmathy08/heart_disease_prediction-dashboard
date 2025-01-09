import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
import joblib  # for saving models
import os
import librosa
import numpy as np

# ===================
# Preprocessing Tabular Data (CSV)
# ===================
def preprocess_tabular_data(file_path, label_encoder=None, scaler=None):
    df = pd.read_csv(file_path)
    print(f"Columns in {file_path}: {df.columns}")  # Debugging: Print out the columns for the dataset

    # Required columns
    required_columns = ['age', 'sex', 'chest_pain', 'trestbps', 'cholesterol']

    # Check if the required columns are in the dataset
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)} in {file_path}")
    
    # Convert 'chest_pain' to numeric using LabelEncoder
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['chest_pain'] = label_encoder.fit_transform(df['chest_pain'])
    else:
        df['chest_pain'] = label_encoder.transform(df['chest_pain'])
    
    # Select only the necessary columns
    X = df[required_columns]  # Features
    y = df['target']  # Target (1: disease, 0: no disease)
    
    # Standardize the features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Convert numpy array to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=required_columns)
    return X_scaled_df, y, scaler, label_encoder

# ===================
# Feature Extraction from Audio Files (MFCC)
# ===================
def extract_features_audio(audio_file):
    y, sr = librosa.load(audio_file)
    features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    return features

# ===================
# Training the Models for All Data Types (Hybrid)
# ===================

# Path to the folder containing CSV files
tabular_folder = r"C:\Users\HONOR\OneDrive\Desktop\FYP 2\data\heart data"  # Update the path to your CSV folder

# List to store features and labels for all files
X_all = []
y_all = []

# Initialize label_encoder and scaler to be reused later
label_encoder = None
scaler = None

# Loop through all files in the folder and process the CSV files
for file_name in os.listdir(tabular_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(tabular_folder, file_name)
        print(f"Processing file: {file_path}")  # Debugging print statement
        try:
            X, y, scaler, label_encoder = preprocess_tabular_data(file_path, label_encoder, scaler)
            X_all.append(X)  # Append DataFrame
            y_all.append(y)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Concatenate data from all CSV files
if X_all and y_all:
    X_all = pd.concat(X_all, ignore_index=True)  # Now it's a DataFrame
    y_all = pd.concat(y_all, ignore_index=True)  # Concatenate labels
else:
    print("No data processed, check file paths and permissions.")

# Verify the shapes of the data before splitting
print(f"X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
if X_all.shape[0] == 0 or y_all.shape[0] == 0:
    print("No valid data available for training!")
else:
    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # Create classifiers for the hybrid model
    svm_model = SVC(kernel="linear", probability=True)
    rf_model = RandomForestClassifier()
    lr_model = LogisticRegression()

    # Train each model
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    # Create the hybrid model (VotingClassifier)
    hybrid_tabular_model = VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model), ('lr', lr_model)], voting='soft')
    hybrid_tabular_model.fit(X_train, y_train)

    # ===================
    # Train Models for Audio Data (Random Forest, SVM, Logistic Regression, Hybrid)
    # ===================
    audio_files_dir = r"C:\Users\HONOR\OneDrive\Desktop\FYP 2\heart prediction dashboard\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training"
    audio_data = []  # List to store extracted features
    audio_labels = []  # List to store labels (1: disease, 0: no disease)

    # Loop through all subdirectories and audio files to extract features
    for subdir, _, files in os.walk(audio_files_dir):  # os.walk() will loop through all subdirectories
        for audio_file in files:
            if audio_file.endswith(('.wav', '.mp3', '.flac')):  # Audio file extensions
                audio_path = os.path.join(subdir, audio_file)
                features = extract_features_audio(audio_path)
                audio_data.append(features)
                audio_labels.append(1 if 'abnormal' in audio_file else 0)  # Assuming filename indicates label

    # Train individual models for Audio
    svm_audio = SVC(kernel="linear", probability=True)
    rf_audio = RandomForestClassifier()
    lr_audio = LogisticRegression()

    # Fit models for Audio data
    svm_audio.fit(audio_data, audio_labels)
    rf_audio.fit(audio_data, audio_labels)
    lr_audio.fit(audio_data, audio_labels)

    # Create hybrid model for Audio data
    hybrid_audio_model = VotingClassifier(estimators=[('svm', svm_audio), ('rf', rf_audio), ('lr', lr_audio)], voting='soft')
    hybrid_audio_model.fit(audio_data, audio_labels)

    # ===================
    # Save all trained models
    # ===================
    os.makedirs('models', exist_ok=True)  # Create 'models' folder if it doesn't exist
    joblib.dump(hybrid_tabular_model, 'models/hybrid_tabular_model.pkl')
    joblib.dump(hybrid_audio_model, 'models/hybrid_audio_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')  # Save the scaler
    joblib.dump(label_encoder, 'models/label_encoder.pkl')  # Save the label encoder

    print("Training complete! All hybrid models are saved.")


