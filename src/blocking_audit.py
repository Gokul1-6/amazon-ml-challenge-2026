# ==========================================
# BLOCKING AUDIT
# ==========================================

import re
import unicodedata
import pandas as pd


# ------------------------------------------
# Unicode-safe text normalization
# ------------------------------------------

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Keep Unicode characters.
    # IMPORTANT:
    # Do NOT convert Hindi/French characters to ASCII.
    value = unicodedata.normalize("NFKC", value)

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ------------------------------------------
# Load dataset
# ------------------------------------------

print("Loading Source 1...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
)

print("Loading Source 2...")

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    dtype=str
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    dtype=str
)


# ------------------------------------------
# Normalize fields
# ------------------------------------------

print("Normalizing Source 1...")

s1["name_norm"] = s1["business_name"].map(normalize_text)
s1["address_norm"] = s1["business_address"].map(normalize_text)

print("Normalizing Source 2...")

s2["name_norm"] = s2["business_name"].map(normalize_text)
s2["address_norm"] = s2["business_address"].map(normalize_text)

print("Normalizing Source 3...")

s3["name_norm"] = s3["business_name"].map(normalize_text)
s3["address_norm"] = s3["business_address"].map(normalize_text)


# ------------------------------------------
# Create blocking keys
# ------------------------------------------

print("Creating blocking keys...")


def add_keys(df):

    # Country + first 8 characters of name
    df["name_key"] = (
        df["country"].fillna("").str.lower()
        + "|"
        + df["name_norm"].str.replace(" ", "", regex=False).str[:8]
    )

    # Country + first 6 characters of address
    df["address_key"] = (
        df["country"].fillna("").str.lower()
        + "|"
        + df["address_norm"].str.replace(" ", "", regex=False).str[:6]
    )

    return df


s1 = add_keys(s1)
s2 = add_keys(s2)
s3 = add_keys(s3)


# ------------------------------------------
# Candidate count using NAME blocking
# ------------------------------------------

print("\n" + "=" * 70)
print("NAME BLOCKING")

name_counts_s2 = (
    s2.groupby("name_key")
    .size()
    .rename("s2_count")
)

name_counts_s3 = (
    s3.groupby("name_key")
    .size()
    .rename("s3_count")
)

s1_name = s1["name_key"]

s1_name_counts = s1_name.map(name_counts_s2).fillna(0)
s1_name_counts += s1_name.map(name_counts_s3).fillna(0)

print("Average candidates per Source-1:")
print(round(s1_name_counts.mean(), 2))

print("Median candidates:")
print(s1_name_counts.median())

print("Maximum candidates:")
print(s1_name_counts.max())


# ------------------------------------------
# Candidate count using ADDRESS blocking
# ------------------------------------------

print("\n" + "=" * 70)
print("ADDRESS BLOCKING")

address_counts_s2 = (
    s2.groupby("address_key")
    .size()
    .rename("s2_count")
)

address_counts_s3 = (
    s3.groupby("address_key")
    .size()
    .rename("s3_count")
)

s1_address = s1["address_key"]

s1_address_counts = (
    s1_address.map(address_counts_s2).fillna(0)
    + s1_address.map(address_counts_s3).fillna(0)
)

print("Average candidates per Source-1:")
print(round(s1_address_counts.mean(), 2))

print("Median candidates:")
print(s1_address_counts.median())

print("Maximum candidates:")
print(s1_address_counts.max())


# ------------------------------------------
# Save simple blocking statistics
# ------------------------------------------

stats = pd.DataFrame({
    "source1_entity_id": s1["entity_id"],
    "name_candidates": s1_name_counts.astype(int),
    "address_candidates": s1_address_counts.astype(int)
})

stats.to_csv(
    "output/blocking_statistics.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 70)
print("DONE")

print("Saved:")
print("output/blocking_statistics.tsv")