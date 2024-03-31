from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


PARENT_DIR = Path(__file__).parent
DATASET_FILE = Path(PARENT_DIR, "data", "formatted_sleep_data.csv")

df = pd.read_csv(DATASET_FILE)

plt.switch_backend("TkAgg")
sns.scatterplot(x="Sedentary Time", y="Stress Level", data=df)
plt.show()
