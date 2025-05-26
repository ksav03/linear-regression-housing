# Linear Regression Housing Price Predictor

This project builds a **Linear Regression** model to predict **median house prices** in California using the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). It is built using **modular Python scripts**, **Jupyter notebooks**, and **Conda environment management**, and is structured for scalability, and ease of automation.

The project serves as a hands-on experience for a complete data science project using Python.


## Features

- **Data Preprocessing** and feature scaling
- **Linear Regression model training**
- **Evaluation with MSE and R²**
- **Prediction on new data**
- **Conda environment setup with `environment.yml`**


## Requirements

All dependencies are managed via Anaconda

## How to use the project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ksav03/linear-regression-housing.git
   cd linear-regression-housing

2. Create the Conda environment using environment file
    ```bash
    conda env create -f environment.yml
    conda activate house-price-env

3. Run the main pipeline - This will download the dataset, preprocess it, train a Linear Regression model, evaluate it, and save the model and scaler.
    ```bash
    python main.py

4. Make predictions on new data - Run the prediction script or use the function from src/predict.py:
    ```bash
    python predict.py

## Author

Keshav Sapkota\
Github: @ksav03



