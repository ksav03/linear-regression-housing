# Using California Housing dataset from sklearn
# However to replicate real-world scenario, we'll load it
# convert it to a CSV file and use the file

import pandas as pd
import os
from sklearn.datasets import fetch_california_housing
from src.config import RAW_DATA_PATH

def save_dataset_as_csv():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH}")


if __name__ == "__main__":
    save_dataset_as_csv()

