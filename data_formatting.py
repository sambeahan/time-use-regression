import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent
DATASET_FILE = Path(
    PARENT_DIR, "regression_training", "data", "sleep_health_dataset.csv"
)


def mins_to_hours(mins: int) -> float:
    return mins / 60


BMI_category_score = {
    "Normal": 21.75,
    "Normal Weight": 21.75,
    "Overweight": 27.5,
    "Obese": 35,
}

# importing data
df = pd.read_csv(DATASET_FILE)
columns_to_drop = [
    "Person ID",
    "Gender",
    "Age",
    "Occupation",
    "Quality of Sleep",
    "Daily Steps",
    "Sleep Disorder",
]
df.drop(columns_to_drop, inplace=True, axis=1)
df["Physical Activity Level"] = df["Physical Activity Level"].apply(mins_to_hours)

df["Sedentary Time"] = 24 - df["Physical Activity Level"] - df["Sleep Duration"]

df[["Systolic Blood Pressure", "Diastolic Blood Pressure"]] = df[
    "Blood Pressure"
].str.split("/", expand=True)

df.drop("Blood Pressure", inplace=True, axis=1)

df["BMI"] = df["BMI Category"].apply(lambda x: BMI_category_score[x])
df.drop("BMI Category", inplace=True, axis=1)

df["Physical Activity Duration"] = df["Physical Activity Level"]
df.drop("Physical Activity Level", inplace=True, axis=1)

print(df.head())
print(df.columns)

df.to_csv(Path(PARENT_DIR, "regression_training", "data", "formatted_sleep_data.csv"))
