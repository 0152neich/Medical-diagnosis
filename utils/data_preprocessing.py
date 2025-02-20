import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path: str, labelencoder_path: str) -> tuple:
    """Load and preprocess data

    Args:
        file_path (str): Path to the dataset
        labelencoder_path (str): Path to save LabelEncoder object
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(file_path)
    df = df.drop(['Outcome Variable'], axis=1)

    # Check for missing values and fix
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)

    # Check for duplicate rows and fix
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    # Encode categorical variables
    df.replace({
        'Fever': {'Yes': 1, 'No': 0},
        'Cough': {'Yes': 1, 'No': 0},
        'Fatigue': {'Yes': 1, 'No': 0},
        'Difficulty Breathing': {'Yes': 1, 'No': 0},
        'Gender': {'Male': 1, 'Female': 0},
        'Blood Pressure': {'Low': 0, 'Normal': 1, 'High': 2},
        'Cholesterol Level': {'Low': 0, 'Normal': 1, 'High': 2},
    }, inplace=True)

    # Encode 'Disease' column
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])

    with open(labelencoder_path, 'wb') as f:
        pickle.dump(le, f)

    # Split data into X and y
    X = df[['Age', 'Gender', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level']]
    y = df['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
