from utils.data_preprocessing import load_and_preprocess_data
from configs.config import DATA_PATH, MODEL_PATH, LABEL_ENCODER_PATH

from sklearn.ensemble import RandomForestClassifier
import pickle

X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH, LABEL_ENCODER_PATH)

model = RandomForestClassifier(n_estimators=40, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Lưu mô hình
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
