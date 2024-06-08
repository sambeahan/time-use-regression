import pickle
import math
from pathlib import Path
import numpy as np
import pandas as pd

PARENT_DIR = Path(__file__).resolve().parent
MODEL_FILE = Path(PARENT_DIR, "models", "time-use-health-2-0.pkl")


def calc_z1(sleep, sedentary, exercise):
    return math.sqrt(2 / 3) * math.log(sleep / math.sqrt(sedentary * exercise))


def calc_z2(sedentary, exercise):
    return (1 / math.sqrt(2)) * math.log(sedentary / exercise)


# load model
with open(MODEL_FILE, "rb") as model_file:
    model = pickle.load(model_file)

while True:
    sleep_time = float(input("Time spent sleeping: "))
    active_time = float(input("Time spent active: "))
    sedentary_time = float(input("Time spent sedentary: "))

    # Apply log ratios if v2.x models
    if str(MODEL_FILE).split("-")[-2] == "2":
        z1 = calc_z1(sleep_time, sedentary_time, active_time)
        z2 = calc_z2(sedentary_time, active_time)

        time_use = [z1, z2, z1 * z1, z1 * z2, z2 * z2]
    else:
        time_use = [sleep_time, active_time, sedentary_time]

    if sum([sleep_time, active_time, sedentary_time]) == 24:
        predictions = model.predict(np.array([time_use]).reshape(1, -1))
        output = pd.DataFrame(
            predictions.T,
            index=[
                "Stress Level",
                "Resting Heart Rate",
                "Systolic Blood Pressure",
                "Diastolic Blood Pressure",
                "BMI",
            ],
        )

        print(output)
    else:
        print("Ensure time use totals to 24 hours.")
    print()
