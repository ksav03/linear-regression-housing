# Step 3: Preprocessing and feature engineering
# Handle missing values, split data into train/test sets, 
# scale features if necessary

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RAW_DATA_PATH

def load_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, scale=True):
    X = df.drop("MedHouseVal", axis=1)  # Features
    y = df["MedHouseVal"]              # Target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None  # In case we need to return it

    return X_train, X_test, y_train, y_test, scaler