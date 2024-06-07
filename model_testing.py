import pickle
from pathlib import Path
import numpy as np
import pandas as pd

PARENT_DIR = Path(__file__).resolve().parent
MODEL_FILE = Path(PARENT_DIR, "models", "time-use-health-2-0.pkl")

# load model
with open(MODEL_FILE, "rb") as model_file:
    model = pickle.load(model_file)

while True:
    sleep_time = float(input("Time spent sleeping: "))
    active_time = float(input("Time spent active: "))
    sedentary_time = float(input("Time spent sedentary: "))
    time_use = [sleep_time, active_time, sedentary_time]

    if sum(time_use) == 24:
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
