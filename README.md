# Auto MPG Prediction using Linear, Ridge, and Lasso Regression  

This project uses the **UCI Auto MPG Dataset** to predict a car’s fuel efficiency (measured in **Miles Per Gallon – MPG**) based on its specifications.  
The project compares **Multiple Linear Regression**, **Ridge Regression**, and **Lasso Regression** models.  

---

## 📌 Project Overview  
- Dataset: [Auto MPG Data (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data)  
- The target variable is **`mpg`** (fuel efficiency).  
- Additional engineered features:  
  - **`power_to_weight`** = horsepower ÷ weight  
  - **`car_age`** = 2025 – model_year  
- Models used:  
  - Multiple Linear Regression (MLR)  
  - Ridge Regression  
  - Lasso Regression  

---

## ⚙️ Steps in the Code  

1. **Import Libraries** → pandas, scikit-learn  
2. **Load Dataset** → UCI Auto MPG dataset  
3. **Data Cleaning** → Removed missing values  
4. **Feature Engineering** → Created `power_to_weight` and `car_age`  
5. **Define Features and Target**  
6. **Train-Test Split** (80% training, 20% testing)  
7. **Feature Scaling** with `StandardScaler`  
8. **Train Models**  
   - Multiple Linear Regression  
   - Ridge Regression  
   - Lasso Regression  
9. **Model Evaluation**  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - R² Score  
10. **Feature Importance** → Print regression coefficients  
11. **Prediction for New Car** → Model predicts MPG for unseen car specifications  

 

