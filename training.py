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
    PARENT_DIR, "regression_training", "data", "formatted_sleep_data.csv"
)

df = pd.read_csv(DATASET_FILE)

plt.switch_backend("TkAgg")


# plotting a scatterplot
"""
sns.scatterplot(x="Sedentary Time", y="Stress Level", data=df)
plt.show()"""

x_vals = df[["Sleep Duration", "Physical Activity Duration", "Sedentary Time"]]
y_labels = [
    "Stress Level",
    "Heart Rate",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "BMI",
]

for y_label in y_labels:
    y = df[y_label]

    print(y_label)

    # print(x_vals)
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        x_vals, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()

    model = model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("mean_squared_error : ", mean_squared_error(y_test, predictions))
    print("mean_absolute_error : ", mean_absolute_error(y_test, predictions))

    print(
        "MAPE",
        mean_absolute_error(y_test, predictions) / df.loc[:, y_label].mean() * 100,
        "%",
    )

    coefficients = pd.concat(
        [pd.DataFrame(x_vals.columns), pd.DataFrame(np.transpose(model.coef_))], axis=1
    )

    print(model.coef_)
