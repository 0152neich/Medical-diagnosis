import pickle
from sklearn.metrics import accuracy_score
from utils.data_preprocessing import load_and_preprocess_data
from configs.config import DATA_PATH, MODEL_PATH, LABEL_ENCODER_PATH

def evaluate_model():
    """ 
        Evaluate the model on test data
    """
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH, LABEL_ENCODER_PATH)

    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    evaluate_model()
