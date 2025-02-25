import streamlit as st
import pickle
import numpy as np
import pandas as pd
from googletrans import Translator
from datetime import timedelta
from configs.config import MODEL_PATH, LABEL_ENCODER_PATH, SUGGESTED_MEDICATION_PATH

# Use the st.cache_data decorator to cache the data
@st.cache_data(ttl=timedelta(hours=1))
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=timedelta(hours=1))
def load_label_encoder():
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=timedelta(hours=1))
def load_suggested_medication():
    return pd.read_csv(SUGGESTED_MEDICATION_PATH)

# Load the model, label encoder, and suggested medication
model = load_model()
le = load_label_encoder()
suggested_medication_df = load_suggested_medication()

st.title("🩺 Hệ thống dự đoán bệnh")
st.write("Vui lòng nhập các thông tin bên dưới để dự đoán bệnh:")

# Use the st.form() context manager to create a form
with st.form(key='diagnosis_form'):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Tuổi", min_value=0, max_value=120, value=25)
        gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
        fever = st.selectbox("Sốt", ["Có", "Không"])
        cough = st.selectbox("Ho", ["Có", "Không"])

    with col2:
        fatigue = st.selectbox("Mệt mỏi", ["Có", "Không"])
        difficulty_breathing = st.selectbox("Khó thở", ["Có", "Không"])
        blood_pressure = st.selectbox("Huyết áp", ["Thấp", "Bình thường", "Cao"])
        cholesterol_level = st.selectbox("Cholesterol", ["Thấp", "Bình thường", "Cao"])

    submit_button = st.form_submit_button(label="🔍 Dự đoán bệnh")

if submit_button:
    gender = 1 if gender == "Nam" else 0
    fever = 1 if fever == "Có" else 0
    cough = 1 if cough == "Có" else 0
    fatigue = 1 if fatigue == "Có" else 0
    difficulty_breathing = 1 if difficulty_breathing == "Có" else 0
    blood_pressure = {"Thấp": 0, "Bình thường": 1, "Cao": 2}[blood_pressure]
    cholesterol_level = {"Thấp": 0, "Bình thường": 1, "Cao": 2}[cholesterol_level]

    # Predict the disease
    input_data = np.array([[age, gender, fever, cough, fatigue, difficulty_breathing, blood_pressure, cholesterol_level]])
    prediction = model.predict(input_data)
    predicted_disease = le.inverse_transform(prediction)[0]

    # Translate the predicted disease to Vietnamese
    translator = Translator()
    predicted_disease_vi = translator.translate(predicted_disease, src='en', dest='vi').text

    # Get the treatment suggestion for the predicted disease
    treatment_suggestion = suggested_medication_df.loc[suggested_medication_df['Disease'] == predicted_disease, 'Treatment Suggestion'].values
    if treatment_suggestion.size > 0:
        treatment_suggestion = treatment_suggestion[0]
        treatment_suggestion_vi = translator.translate(treatment_suggestion, src='en', dest='vi').text
    else:
        treatment_suggestion_vi = "Không có gợi ý điều trị cho bệnh này."

    # Show the prediction results
    st.subheader("🔔 Kết quả dự đoán:")
    st.write(f"💊 **Bệnh có thể mắc phải:** {predicted_disease_vi}")
    st.write(f"🩹 **Gợi ý điều trị:** {treatment_suggestion_vi}")
    st.warning("📌 Lưu ý: Đây chỉ là dự đoán, không thay thế cho việc thăm khám y tế chuyên nghiệp.")
