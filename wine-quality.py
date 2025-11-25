import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

from ucimlrepo import fetch_ucirepo 

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

X = wine_quality.data.features 
y = wine_quality.data.targets 

# metadata 
print(wine_quality.metadata) 

# variable information 
print(wine_quality.variables) 




df = wine_quality.data.original

# Remove red wine data
df = df[df['color'] == 'white']

# Drop the 'color' column as it's no longer needed
df.drop(columns=['color'], inplace=True)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Define function for model evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Model': model.__class__.__name__,
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred),
        'R^2': r2_score(y_test, y_pred),
    }

# Linear Regression
lr = LinearRegression()

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)

# XGBoost Regressor
xg = XGBRegressor(random_state=42)

models = [lr, dt, rf, gb, xg]
results = []
for model in models:
    result = evaluate_model(model, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())
    results.append(result)



# Use the two best performing base models for tuning (Random Forest and XGBoost)
# Hyperparameter tuning for Random Forest and XGBoost

param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 5, 8]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error',n_jobs=1)
rf_cv = grid_search.fit(X_train, y_train.values.ravel())

param_grid_xg = {
    'n_estimators': [250, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

xg_grid_search = GridSearchCV(estimator=xg, param_grid=param_grid_xg, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
xg_cv = xg_grid_search.fit(X_train, y_train)



# Results of the best models
best_rf = rf_cv.best_estimator_
best_xg = xg_cv.best_estimator_

# Evaluate the best models
rf_result = evaluate_model(best_rf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())
xg_result = evaluate_model(best_xg, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())


# Save model
joblib.dump(best_xg, "wine_model.joblib")
