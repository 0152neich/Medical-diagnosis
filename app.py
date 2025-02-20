import streamlit as st
import pickle
import numpy as np
from configs.config import MODEL_PATH, LABEL_ENCODER_PATH
from googletrans import Translator

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load LabelEncoder to convert predicted label to disease name
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

st.title("🩺 Hệ thống dự đoán bệnh")
st.write("Vui lòng nhập các thông tin bên dưới để dự đoán bệnh:")
col1, col2 = st.columns([1, 1])

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

# Preprocess input data
gender = 1 if gender == "Nam" else 0
fever = 1 if fever == "Có" else 0
cough = 1 if cough == "Có" else 0
fatigue = 1 if fatigue == "Có" else 0
difficulty_breathing = 1 if difficulty_breathing == "Có" else 0
blood_pressure = {"Thấp": 0, "Bình thường": 1, "Cao": 2}[blood_pressure]
cholesterol_level = {"Thấp": 0, "Bình thường": 1, "Cao": 2}[cholesterol_level]

# Predict disease
if st.button("🔍 Dự đoán bệnh"):
    input_data = np.array([[age, gender, fever, cough, fatigue, difficulty_breathing, blood_pressure, cholesterol_level]])
    prediction = model.predict(input_data)
    predicted_disease = le.inverse_transform(prediction)[0]
    # Translate predicted disease to Vietnamese
    translator = Translator()
    predicted_disease = translator.translate(predicted_disease, src='en', dest='vi').text

    st.subheader("🔔 Kết quả dự đoán:")
    st.write(f"💊 Bệnh có thể mắc phải: **{predicted_disease}**")
