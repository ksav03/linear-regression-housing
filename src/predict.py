import joblib
import numpy as np
from config import MODEL_DIR

def load_model():
    model = joblib.load(f"{MODEL_DIR}/linear_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return model, scaler

def predict_new(data: np.ndarray):
    model, scaler = load_model()
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    return predictions

if __name__ == "__main__":
    # Example: predict for 2 samples (same shape/order as training data)
    sample_data = np.array([
        [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23],
        [5.6431, 34.0, 6.2381, 0.9891, 456.0, 2.345, 36.77, -121.88]
    ])
    results = predict_new(sample_data)
    print("Predicted prices:", results)