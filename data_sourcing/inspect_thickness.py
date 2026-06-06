import pandas as pd
from pathlib import Path

# Replace with your filenames
CRUST_FILE = "crustal_thickness.xyz"
SEDIMENT_FILE = "sediment_thickness.xyz"


def inspect_xyz(file_path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path}")
    print("=" * 80)

    path = Path(file_path)

    print(f"\nExists: {path.exists()}")
    print(f"Size: {path.stat().st_size / (1024*1024):.2f} MB")

    # Read first few lines raw
    print("\nFIRST 10 RAW LINES:")
    with open(file_path, "r") as f:
        for i in range(10):
            line = f.readline()
            if not line:
                break
            print(repr(line.strip()))

    print("\nTrying whitespace-separated read...")

    try:
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,
            header=None
        )

        print("\nShape:")
        print(df.shape)

        print("\nFirst 10 rows:")
        print(df.head(10))

        print("\nColumn statistics:")
        print(df.describe())

        print("\nNumber of columns:")
        print(df.shape[1])

        if df.shape[1] >= 3:
            print("\nPotential coordinate ranges:")

            for col in range(min(3, df.shape[1])):
                print(
                    f"Column {col}: "
                    f"min={df[col].min()}, "
                    f"max={df[col].max()}"
                )

            print("\nUnique values in first column (sample):")
            print(df[0].unique()[:10])

            print("\nUnique values in second column (sample):")
            print(df[1].unique()[:10])

    except Exception as e:
        print("\nERROR READING FILE:")
        print(e)


inspect_xyz(CRUST_FILE)
inspect_xyz(SEDIMENT_FILE)
