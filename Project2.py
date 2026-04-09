import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df.head())

# Pre-processing Techniques:

# remove the id column
df = df.drop(columns=["id"])
# fill missing values in the bmi column with the mean
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
# converted categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Regression Techniques

# Majority Class Classifier:

# pre-processing the data for regression
df_reg = pd.read_csv("healthcare-dataset-stroke-data.csv")
df_reg = df_reg.drop(columns=["id"])
df_reg = df_reg.dropna(subset=["bmi"])
df_reg = pd.get_dummies(df_reg, drop_first=True)

X = df_reg.drop(columns=["bmi"])
y = df_reg["bmi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
y_pred = dummy_regr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nDummy Regressor Results:")
print("RMSE:", rmse)
print("MAE:", mae)