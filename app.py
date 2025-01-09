import streamlit as st
import pandas as pd
import joblib
import numpy as np
from feature_extraction.feature_extraction_audio import extract_features_audio
import base64
from fpdf import FPDF
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load trained models
svm_model = joblib.load('models/svm_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
hybrid_model = joblib.load('models/hybrid_model.pkl')
scaler = joblib.load('models/scaler.pkl')
audio_model = joblib.load('models/audio_model.pkl')

# Function to make predictions
def predict_heart_disease(model, features):
    return model.predict(features)[0]

# Function to display metrics
def display_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    st.subheader("Performance Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    st.write("Classification Report:")
    st.write(class_report)
st.set_page_config(layout="wide")
# Provide Health Recommendations
def provide_recommendations(prediction):
    """
    Provide actionable recommendations based on the prediction.
    """
    if prediction == 1:
        st.warning("**Actionable Recommendations:**")
        st.markdown("- Consult a cardiologist immediately.")
        st.markdown("- Monitor your cholesterol and blood pressure levels.")
        st.markdown("- Adopt a heart-healthy diet and increase physical activity.")
    else:
        st.success("**Keep up the good work!**")
        st.markdown("- Maintain a healthy lifestyle with regular exercise.")
        st.markdown("- Avoid smoking and excessive alcohol consumption.")
        st.markdown("- Get regular health checkups.")

# Save Results as PDF with a title
def save_as_pdf(input_type, data, prediction, probabilities):
    pdf = FPDF()
    pdf.add_page()
    
    # Title for the PDF
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt=f"{input_type} - Heart Disease Prediction Results", ln=True, align='C')
    
    # Add some space
    pdf.ln(10)
    
    # Set font for content
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"Input Type: {input_type}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Input Data: {data}", ln=True)
    pdf.ln(5)
    
    # Display the prediction result
    prediction_text = "Heart Disease" if prediction == 1 else "No Heart Disease"
    pdf.cell(200, 10, txt=f"Prediction: {prediction_text}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Probabilities: No Heart Disease: {probabilities[0]:.2f}, Heart Disease: {probabilities[1]:.2f}", ln=True)
    
    # Output PDF file
    pdf_file = f"prediction_result_{input_type.replace(' ', '_')}.pdf"
    pdf.output(pdf_file)
    
    st.success(f"Results saved as {pdf_file}.")
    with open(pdf_file, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=pdf_file, mime="application/pdf")

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background("heart_background.jpeg")

# Create columns for layout: Left panel, Middle panel (input section), and Right panel (additional info)
col1, col2, col3 = st.columns([1, 4, 1])  

# Left Panel with Heart Disease Info
with col1:
    st.markdown(
        """
        <style>
            .info-box {
                background-color: rgba(255, 255, 255, 0.8);
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                margin: 20px 0;
            }
            .left-panel {
                position: fixed;
                top: 10%;
                left: 0;
                width: 300px;
                height: 80%;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                z-index: 1000;
                overflow-y: scroll;
            }
            .main-content {
                margin-left: 320px; /* Adjust for the left panel */
                padding: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="left-panel">
            <img src="cholesterol_chart.jpg" alt="Cholesterol Chart" style="width:100%; margin-bottom:20px;">
            <h3>About Heart Disease</h3>
            <p>Heart disease includes conditions like coronary artery disease, arrhythmias, and congenital heart defects.</p>
            <h4><strong>Key Causes:</strong></h4>
            <ul>
                <li>High blood pressure</li>
                <li>High cholesterol</li>
                <li>Smoking</li>
                <li>Diabetes</li>
                <li>Obesity</li>
            </ul>
            <h4><strong>Symptoms:</strong></h4>
            <ul>
                <li>Chest pain or discomfort</li>
                <li>Shortness of breath</li>
                <li>Fatigue</li>
                <li>Irregular heartbeat</li>
            </ul>
            <h4><strong>Preventive Tips:</strong></h4>
            <ul>
                <li>Stay active and eat a balanced diet.</li>
                <li>Avoid smoking and excessive alcohol.</li>
                <li>Monitor blood pressure and cholesterol levels.</li>
                <li>Manage stress and get regular health checkups.</li>
            </ul>
            <p><strong>Early detection can save lives.</strong> 💖</p>
        </div>
        <style>
            .left-panel {{
                padding: 10px;
                position: sticky;
                top: 0;
            }}
            .left-panel h3 {{
                font-size: 1.5em;
            }}
            .left-panel h4 {{
                font-size: 1.2em;
            }}
            .left-panel p {{
                font-size: 1em;
            }}
            .left-panel ul {{
                font-size: 1em;
                margin-left: 20px;
            }}
            .left-panel ul li {{
                margin-bottom: 5px;
            }}
           }}
            .main-content {{
                margin-left: 320px; /* Adjust for the left panel */
                padding: 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Middle Panel (Main Input Section)
with col2:
    st.title("🫀Heart Disease Prediction Dashboard")
    st.header("Select Input Type for Heart Disease Prediction")

    input_type = st.radio("Choose Input Type", ['Manual Data', 'Audio File'])

    if input_type == 'Manual Data':
        st.subheader("Enter Patient Data for Heart Disease Prediction")
        
         # Collapsible section for angina information
        with st.expander("What is Angina?"):
                st.markdown("""
                ### Types of Angina
                #### 1. Typical Angina
                - **Definition**: Chest pain or discomfort due to insufficient blood flow to the heart (ischemia).
                - **Symptoms**:
                    - **Location**: Central or left-sided chest pain, often described as "pressure" or "tightness."
                    - **Radiation**: Pain may radiate to shoulders, arms (often left), neck, jaw, or back.
                    - **Trigger**: Occurs with physical exertion, emotional stress, or after meals.
                    - **Duration**: Typically lasts 5–15 minutes.
                    - **Relief**: Improves with rest or nitroglycerin.
                - **Associated Symptoms**: 
                    - Shortness of breath, sweating (diaphoresis), nausea, fatigue.

                #### 2. Atypical Angina
                - **Definition**: Symptoms that do not fully match the classic presentation of angina. Common in women, elderly, and diabetics.
                - **Symptoms**:
                    - **Pain Characteristics**: Burning, sharp, or stabbing pain instead of pressure or tightness.
                    - **Location**: May not involve the chest; could occur in the back, neck, jaw, or abdomen.
                    - **Trigger**: Less associated with physical exertion; may occur at rest.
                    - **Associated Symptoms**: Indigestion-like sensation, fatigue, dizziness, dyspnea.

                #### 3. Non-Anginal Pain
                - **Definition**: Chest pain unrelated to cardiac ischemia.
                - **Symptoms**:
                    - **Pain Characteristics**: Sharp, stabbing, or localized pain.
                    - **Trigger**: Pain worsens with movement, breathing, or palpation.
                    - **Duration**: Often lasts seconds to hours.
                    - **Relief**: Unpredictable; may not respond to rest or nitroglycerin.
                    - **Causes**: Musculoskeletal pain, GERD, pulmonary issues, anxiety, or panic attacks.

                #### 4. Asymptomatic
                - **Definition**: Lack of typical symptoms despite ischemia or heart disease.
                - **Subtle Symptoms (individuals may feel**:
                    - Unexplained fatigue, mild breathlessness, and general discomfort or unease.
                - **Absence of Chest Pain**: No noticeable discomfort or warning signs.
                - **Common in**:
                    - Diabetics (due to neuropathy), elderly individuals, post-heart transplant patients.
                """)
        age = st.number_input("Age", min_value=0)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        chest_pain_type = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        cholesterol = st.number_input("Cholesterol Level", min_value=0)

        sex = 1 if sex == 'Male' else 0
        chest_pain_type = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(chest_pain_type)

        tabular_input = pd.DataFrame([[age, sex, chest_pain_type, trestbps, cholesterol]], 
                                     columns=['age', 'sex', 'chest_pain', 'trestbps', 'cholesterol'])
        tabular_input_scaled = scaler.transform(tabular_input)

        if st.button("Predict"):
            prediction = predict_heart_disease(hybrid_model, tabular_input_scaled)
            prediction_probabilities = hybrid_model.predict_proba(tabular_input_scaled)[0]
            if prediction == 1:
                st.error("Prediction: **Heart Disease Detected**")
            else:
                st.success("Prediction: **No Heart Disease Detected**")
            provide_recommendations(prediction)
            save_as_pdf("Manual Input", f"Age: {age}, Sex: {'Male' if sex == 1 else 'Female'}, Chest Pain Type: {chest_pain_type}, Trestbps: {trestbps}, Cholesterol: {cholesterol}", prediction, prediction_probabilities)

    elif input_type == 'Audio File':
        st.subheader("Upload Audio File for Prediction")
        uploaded_audio = st.file_uploader("Choose an Audio File", type=["wav", "mp3", "ogg"])

        if uploaded_audio is not None:
            audio_features = extract_features_audio(uploaded_audio)
            st.audio(uploaded_audio, format='audio/wav')

            if st.button("Predict"):
                prediction_audio = predict_heart_disease(audio_model, [audio_features])
                prediction_probabilities = audio_model.predict_proba([audio_features])[0]
                if prediction_audio == 1:
                    st.error("Prediction: **Abnormal Heart Sound**")
                else:
                    st.success("Prediction: **Normal Heart Sound**")
                provide_recommendations(prediction_audio)
                save_as_pdf("Audio File", f"Audio File: {uploaded_audio.name}", prediction_audio, prediction_probabilities)

# Right Panel (Additional Info Section)
with col3:
    st.markdown("---")
    st.markdown(
        """
        <div class="info-section">
            <p><strong>More Information:</strong></p>
            <p>For more details, visit the <a href="https://www.ijn.com.my" target="_blank" style="color: blue; text-decoration: underline;">Institut Jantung Negara</a>.</p>
            <div class="info-box">
                <ul>
                    <li><strong>Heart Sounds:</strong> Abnormal heart sounds (murmurs) can indicate conditions like valve defects or heart failure.</li>
                    <li><strong>ECG (Electrocardiogram):</strong> Abnormal ECG readings can reveal arrhythmias or other cardiac problems.</li>
                </ul>
                <p>Early diagnosis and monitoring are crucial for effective treatment and management of heart diseases.</p>
            </div>
        </div>
        <style>
            .info-section {
                background-color: #f0f8ff; /* Light blue background */
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                margin-top: 20px;
            }
            .info-section p, .info-section ul {
                font-size: 1em;
                color: #333;
            }
            .info-box ul li {
                margin-bottom: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Add embedded video for heart disease
    st.subheader(" Video")
    video_url = "https://www.youtube.com/watch?v=lTCF8y7e1Bw"
    st.video(video_url)
