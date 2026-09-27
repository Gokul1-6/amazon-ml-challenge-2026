# Pandas library
import pandas as pd


# -----------------------------------------
# Function to inspect country distribution
# -----------------------------------------

def country_report(file):

    print("\n" + "=" * 70)
    print("COUNTRY REPORT:", file)

    df = pd.read_csv(
        file,
        sep="\t",
        dtype=str,
        usecols=["country"]
    )

    print("\nCountry distribution:")
    print(df["country"].value_counts(dropna=False))


# -----------------------------------------
# Function to inspect missing values
# -----------------------------------------

def missing_report(file):

    print("\n" + "=" * 70)
    print("MISSING VALUES:", file)

    df = pd.read_csv(
        file,
        sep="\t",
        dtype=str
    )

    print(df.isna().sum())


# -----------------------------------------
# Country reports
# -----------------------------------------

country_report("dataset/train/train_source1.tsv")
country_report("dataset/train/train_source2.tsv")
country_report("dataset/train/train_source3.tsv")

country_report("dataset/test/test_source1.tsv")
country_report("dataset/test/test_source2.tsv")
country_report("dataset/test/test_source3.tsv")


# -----------------------------------------
# Ground truth analysis
# -----------------------------------------

print("\n" + "=" * 70)
print("GROUND TRUTH MATCH ANALYSIS")

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Empty match field = zero matches
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if not x.strip() else len(x.split(","))
)

print("\nMatch count distribution:")
print(gt["match_count"].value_counts().sort_index())

print("\nMaximum matches for one Source-1:")
print(gt["match_count"].max())

print("\nZero-match Source-1 records:")
print((gt["match_count"] == 0).sum())

print("\nOne-match Source-1 records:")
print((gt["match_count"] == 1).sum())

print("\nMultiple-match Source-1 records:")
print((gt["match_count"] > 1).sum())