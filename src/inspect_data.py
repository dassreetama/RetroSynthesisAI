
from pathlib import Path
import pandas as pd

data_dir = Path("data")

for name in ["raw_train.csv", "raw_val.csv", "raw_test.csv"]:
    file = data_dir / name

    print("\n" + "=" * 50)
    print("FILE:", name)

    df = pd.read_csv(file)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
