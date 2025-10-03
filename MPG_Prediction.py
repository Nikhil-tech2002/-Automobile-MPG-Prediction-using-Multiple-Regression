# 1. Import Libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
df = pd.read_csv(url, names=columns, sep='\s+', na_values='?') 


#  3. Data Cleaning
df = df.dropna() 

#  4. Feature Engineering 
df['power_to_weight'] = df['horsepower'] / df['weight']
df['car_age'] = 2025 - df['model_year']

# 5. Define Features and Target
features = ['cylinders','displacement','horsepower','weight','acceleration','model_year','power_to_weight','car_age']
X = df[features]
y = df['mpg']

#  6. Split Dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#  7. Scale Features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Models 
# Multiple Linear Regression
mlr_model = LinearRegression()
mlr_model.fit(X_train_scaled, y_train)
y_pred_mlr = mlr_model.predict(X_test_scaled)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Lasso Regression
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

#  9. Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_true, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_true, y_pred))
    print("R² Score:", r2_score(y_true, y_pred))

evaluate_model(y_test, y_pred_mlr, "Multiple Linear Regression")
evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_lasso, "Lasso Regression")
# 10. Feature Importance 
print("\nMLR Coefficients:")
for feature, coef in zip(features, mlr_model.coef_):
    print(f"{feature}: {coef}")

#  11. Predict MPG for a New Car 
new_car = pd.DataFrame({
    'cylinders':[4], 'displacement':[150], 'horsepower':[95], 'weight':[2600],
    'acceleration':[15], 'model_year':[82], 'power_to_weight':[95/2600], 'car_age':[2025-82]
})
new_car_scaled = scaler.transform(new_car)
predicted_mpg = mlr_model.predict(new_car_scaled)
print("\nPredicted MPG for new car:", predicted_mpg[0])
