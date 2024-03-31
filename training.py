"""Trains a linear regression model on the formatted dataset"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

PARENT_DIR = Path(__file__).parent.parent
DATASET_FILE = Path(PARENT_DIR, "data", "formatted_sleep_data.csv")

df = pd.read_csv(DATASET_FILE)


X = df[["Sleep Duration", "Physical Activity Duration", "Sedentary Time"]]
Y = df[
    [
        "Stress Level",
        "Heart Rate",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "BMI",
    ]
]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

model = LinearRegression(fit_intercept=False)

model = model.fit(X_train.values, y_train.values)


# print(pd.DataFrame(zip(X_train.columns, model.coef_)))
coefficients = pd.DataFrame(model.coef_, index=Y.columns, columns=X.columns)
# print(f"[{X_train.columns}]\n [{model.coef_.T}] + {model.intercept_}")
print("Model coefficients:")
print(coefficients)

print("\nModel intercepts:")
if type(model.intercept_).__module__ == np.__name__:
    intercepts = pd.DataFrame(model.intercept_, index=Y.columns)
    print(intercepts)
else:
    print(model.intercept_)

print("\nModel accuracy:")
for i in range(len(Y.columns)):
    column = Y.columns[i]
    print(column)
    predictions = model.predict(X_test.values)[:, i : i + 1]

    print("mean_squared_error : ", mean_squared_error(y_test[column], predictions))
    print("mean_absolute_error : ", mean_absolute_error(y_test[column], predictions))

    print(
        "MAPE",
        mean_absolute_error(y_test[column], predictions)
        / df.loc[:, column].mean()
        * 100,
        "%\n",
    )
