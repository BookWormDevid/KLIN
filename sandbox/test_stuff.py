from pathlib import Path

import pandas as pd

path = Path("video_classification_results.csv")

df = pd.read_csv(path)

print(df["predicted_class"].value_counts())
