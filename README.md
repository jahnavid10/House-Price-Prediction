# 🏡 Housing Price Prediction Project

This repository contains a complete end-to-end machine learning pipeline for predicting housing prices based on various features. The project is organized into four structured Jupyter notebooks, covering data loading, cleaning, feature engineering, model training, evaluation, and comparison. The dataset used is `housing.csv`.

---

## 📁 Project Structure

```
├── housing.csv                             # Raw dataset used for training and evaluation
├── 01_data_loading_and_cleaning.ipynb      # Step 1: Load and clean the data
├── 02_feature_engineering.ipynb            # Step 2: Engineer features
├── 03_model_training.ipynb                 # Step 3: Train ML models
├── 04_model_evaluation_and_comparison.ipynb # Step 4: Evaluate and compare models
└── README.md                               # Project documentation
```

---

## 📊 Dataset

The dataset (`housing.csv`) consists of housing information including:
- Numerical features (e.g., number of rooms, bedrooms, population)
- Categorical features (e.g., location)
- Target variable: **Median house value**

---

## 🔍 Step-by-Step Notebooks

### 1. 📂 `01_data_loading_and_cleaning.ipynb`
- Loads the housing dataset.
- Performs initial exploration and visualization.
- Cleans missing data.
- Handles outliers and invalid values.
- Provides data summary and insights.

### 2. 🛠 `02_feature_engineering.ipynb`
- Creates new features from raw data (e.g., `rooms_per_household`, `bedrooms_per_room`).
- Applies encoding to categorical features (e.g., OneHotEncoding).
- Normalizes or standardizes numerical data.
- Prepares data for modeling.

### 3. 🤖 `03_model_training.ipynb`
- Splits the data into training and test sets.
- Trains multiple machine learning models (e.g., Linear Regression, Decision Tree, Random Forest).
- Uses cross-validation to tune hyperparameters.
- Logs model training performance.

### 4. 📈 `04_model_evaluation_and_comparison.ipynb`
- Evaluates model performance using RMSE and R².
- Compares models visually and numerically.
- Selects the best model based on evaluation metrics.
- Optionally saves the trained model for deployment.

---

## 🧰 Requirements

To run the notebooks, install the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/housing-price-prediction.git
   cd housing-price-prediction
   ```

2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

3. Open each notebook in sequence from `01_` to `04_` and run the cells step by step.

---

## 📌 Key Learnings

- Practical implementation of an end-to-end ML workflow
- Importance of EDA and feature engineering
- Model comparison and hyperparameter tuning
- Modular and reproducible notebook design

---

