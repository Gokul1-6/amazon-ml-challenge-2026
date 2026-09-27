# Pandas library import
import pandas as pd


# Dataset files
files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]


# Check every dataset
for file in files:

    print("\n" + "=" * 70)
    print("FILE:", file)

    # Read TSV file
    df = pd.read_csv(
        file,
        sep="\t",
        dtype=str
    )

    # Basic information
    print("Rows    :", len(df))
    print("Columns :", list(df.columns))
    print("Missing :", df.isna().sum().sum())

    # Show first 3 records
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))