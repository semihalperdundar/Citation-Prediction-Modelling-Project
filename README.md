# Citation Prediction Modeling

This repository contains a **Machine Learning Model** designed to predict the number of citations a scientific paper will receive based on its metadata. The model is built using **Python, Scikit-learn, and XGBoost**, incorporating advanced **feature engineering** and **hyperparameter tuning** to optimize predictive accuracy.

## 📂 Project Structure

```
├── Citation_Prediction_Modelling_ML.py
├── README.md
```

### 📌 **Files Overview**

- **Citation_Prediction_Modelling_ML.py** → The main Python script implementing feature engineering, model selection, training, and prediction.
- **description.docx** → Detailed explanation of the model pipeline, evaluation metrics, and optimization techniques.
- **README.md** → This document.

## 🎯 Objective

The goal of this project is to **develop a citation prediction model** using **metadata-based features** extracted from academic papers. The model predicts future citation counts using various machine learning techniques and optimizations.

## 🔬 **Methodology**

### **Feature Engineering**
The model extracts and processes the following features:
- **Numerical Features**:
  - Year of publication
  - Number of authors
  - Number of references cited
  - Paper age (difference between publication year and current year)
  - Title word count
- **Categorical Features**:
  - Venue (converted into numerical format using Label Encoding)
- **Text Features**:
  - Title and abstract transformed using **TF-IDF vectorization**

### **Model Selection & Training**
The project evaluates several models to determine the best approach for citation prediction:
- **Ridge Regression** (Regularized linear model)
- **Gradient Boosting** (Ensemble learning)
- **XGBoost** (Optimized gradient boosting)
- **Random Forest** (Best-performing model)
- **Ensemble Model** (Combining multiple models for improved accuracy)

### **Optimization & Validation**
- **Hyperparameter tuning**: Conducted using **RandomizedSearchCV** and **GridSearchCV**
- **Cross-validation**: Applied to improve generalization and avoid overfitting
- **Log transformation**: Citation counts were log-transformed for better model learning

## 🛠️ Installation & Setup

To run the model, install the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost sentence-transformers torch
```

## 🚀 Running the Model

To execute the citation prediction model:
```bash
python Citation_Prediction_Modelling_ML.py
```

The script will:
1. **Load the training dataset**
2. **Perform feature engineering**
3. **Train multiple models**
4. **Evaluate models using Mean Absolute Error (MAE)**
5. **Select the best-performing model**
6. **Generate predictions for test data**

## 📊 Results & Performance
- **Best Model**: **Random Forest Regressor & Ensemble Model**
- **Validation Mean Absolute Error (MAE)**: **31.27**
- **Hyperparameters Tuned**: `n_estimators=200`, `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=0.8`

## 🔍 Key Findings
- **TF-IDF vectorization of abstract and title significantly improves performance**
- **Random Forest outperforms Gradient Boosting, Ridge, and XGBoost in citation prediction**
- **Ensemble modeling further refines predictions by combining different model strengths**
- **Cross-validation helps in preventing overfitting and improves model robustness**

## 📌 Future Improvements
- Experiment with **deep learning models** like Transformer-based architectures (BERT, SciBERT)
- Introduce additional **contextual metadata**, such as journal impact factors
- Enhance **feature engineering** with network-based features (e.g., citation graphs)

## 👨‍💻 Author
- **Semih Alper Dundar**  
- **Tilburg University - Data Science & Society**

## 📝 License
This project is released under the **MIT License**.

---

This repository contributes to **research on scientific impact prediction**, helping to understand **factors influencing paper citations** and improving data-driven decision-making in academia. 📚📈
