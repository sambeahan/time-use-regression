"""Formats data for training"""

from pathlib import Path
import pandas as pd

PARENT_DIR = Path(__file__).parent
INPUT_DATASET_FILE = Path(PARENT_DIR, "data", "sleep_health_dataset.csv")
OUTPUT_DATASET_FILE = Path(PARENT_DIR, "data", "formatted_sleep_data.csv")


def mins_to_hours(mins: int) -> float:
    """Converts minutes to hours"""
    return mins / 60


BMI_category_score = {
    "Normal": 21.75,
    "Normal Weight": 21.75,
    "Overweight": 27.5,
    "Obese": 35,
}

# importing data
df = pd.read_csv(INPUT_DATASET_FILE)
df["Physical Activity Level"] = df["Physical Activity Level"].apply(mins_to_hours)

df["Sedentary Time"] = 24 - df["Physical Activity Level"] - df["Sleep Duration"]

df[["Systolic Blood Pressure", "Diastolic Blood Pressure"]] = df[
    "Blood Pressure"
].str.split("/", expand=True)

df["BMI"] = df["BMI Category"].apply(lambda x: BMI_category_score[x])

df["Physical Activity Duration"] = df["Physical Activity Level"]

df["Resting Heart Rate"] = df["Heart Rate"]


columns_to_drop = [
    "Person ID",
    "Gender",
    "Age",
    "Occupation",
    "Quality of Sleep",
    "Daily Steps",
    "Sleep Disorder",
    "Blood Pressure",
    "BMI Category",
    "Physical Activity Level",
    "Heart Rate",
]
df.drop(columns_to_drop, inplace=True, axis=1)

print(df.head())

df.to_csv(OUTPUT_DATASET_FILE)
