import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


# load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df.head())

# Preprocessing the data:

# Remove the id column
df = df.drop(columns=["id"])
#fill missing values in the bmi column with the mean
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
# converted categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)