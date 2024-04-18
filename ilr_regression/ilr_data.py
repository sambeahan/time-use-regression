"""Adds the ILR balances to the dataset"""

import math
from pathlib import Path
import pandas as pd

PARENT_DIR = Path(__file__).parent.parent
INPUT_DATASET_FILE = Path(PARENT_DIR, "data", "formatted_sleep_data.csv")
OUTPUT_DATASET_FILE = Path(PARENT_DIR, "data", "ilr_sleep_data.csv")


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

z1_vals = []
z2_vals = []

for count, row in df.iterrows():
    z1_vals.append(
        math.sqrt(2 / 3)
        * math.log(
            row["Sleep Duration"]
            / math.sqrt(row["Sedentary Time"] * row["Physical Activity Duration"])
        )
    )

    z2_vals.append(
        (1 / math.sqrt(2))
        * math.log(row["Sedentary Time"] / row["Physical Activity Duration"])
    )

df["z1"] = z1_vals
df["z2"] = z2_vals

# Add combination rows
df["z1z1"] = df["z1"] * df["z1"]
df["z1z2"] = df["z1"] * df["z2"]
df["z2z2"] = df["z2"] * df["z2"]


"""
df["Sedentary Time"] = 24 - df["Physical Activity Level"] - df["Sleep Duration"]

df[["Systolic Blood Pressure", "Diastolic Blood Pressure"]] = df[
    "Blood Pressure"
].str.split("/", expand=True)

df["BMI"] = df["BMI Category"].apply(lambda x: BMI_category_score[x])

df["Physical Activity Duration"] = df["Physical Activity Level"]

df["Resting Heart Rate"] = df["Heart Rate"]



"""

columns_to_drop = ["Sleep Duration", "Sedentary Time", "Physical Activity Duration"]
df.drop(columns_to_drop, inplace=True, axis=1)
print(df.head())

df.to_csv(OUTPUT_DATASET_FILE)
