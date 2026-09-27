# ==========================================
# BLOCKING RECALL AUDIT
# ==========================================

import re
import unicodedata
import pandas as pd


# ------------------------------------------
# Unicode-safe normalization
# ------------------------------------------

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Keep Hindi / French / other Unicode text
    value = unicodedata.normalize("NFKC", value)

    # Replace punctuation
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ------------------------------------------
# Load data
# ------------------------------------------

print("Loading datasets...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)


# ------------------------------------------
# Normalize
# ------------------------------------------

print("Normalizing...")

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].map(
        normalize_text
    )

    df["address_norm"] = df["business_address"].map(
        normalize_text
    )


# ------------------------------------------
# Create blocking keys
# ------------------------------------------

def add_keys(df):

    df["name_key"] = (
        df["country"].fillna("").str.lower()
        + "|"
        + df["name_norm"]
        .str.replace(" ", "", regex=False)
        .str[:8]
    )

    df["address_key"] = (
        df["country"].fillna("").str.lower()
        + "|"
        + df["address_norm"]
        .str.replace(" ", "", regex=False)
        .str[:6]
    )

    return df


s1 = add_keys(s1)
s2 = add_keys(s2)
s3 = add_keys(s3)


# ------------------------------------------
# Create candidate sets
# ------------------------------------------

print("Creating blocking indexes...")

name_index = {}
address_index = {}


# Source 2 + Source 3
for df in [s2, s3]:

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        name_key = row["name_key"]
        address_key = row["address_key"]

        if name_key:
            name_index.setdefault(
                name_key, set()
            ).add(entity_id)

        if address_key:
            address_index.setdefault(
                address_key, set()
            ).add(entity_id)


# ------------------------------------------
# Ground truth lookup
# ------------------------------------------

truth = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    raw = row["matched_entity_ids"].strip()

    if raw:
        truth[s1_id] = set(
            x.strip()
            for x in raw.split(",")
            if x.strip()
        )
    else:
        truth[s1_id] = set()


# ------------------------------------------
# Recall calculation
# ------------------------------------------

total_true_matches = 0

name_captured = 0

address_captured = 0

union_captured = 0


print("Checking ground-truth matches...")


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    true_matches = truth.get(
        s1_id,
        set()
    )

    if not true_matches:
        continue

    total_true_matches += len(true_matches)

    name_candidates = name_index.get(
        row["name_key"],
        set()
    )

    address_candidates = address_index.get(
        row["address_key"],
        set()
    )

    name_captured += len(
        true_matches & name_candidates
    )

    address_captured += len(
        true_matches & address_candidates
    )

    union_candidates = (
        name_candidates |
        address_candidates
    )

    union_captured += len(
        true_matches & union_candidates
    )


# ------------------------------------------
# Final results
# ------------------------------------------

print("\n" + "=" * 70)

print("TOTAL TRUE MATCHES:")
print(total_true_matches)

print("\nNAME BLOCKING CAPTURED:")
print(name_captured)

print("\nADDRESS BLOCKING CAPTURED:")
print(address_captured)

print("\nUNION BLOCKING CAPTURED:")
print(union_captured)


if total_true_matches > 0:

    print("\n" + "=" * 70)

    print(
        "NAME BLOCKING RECALL:",
        round(
            name_captured / total_true_matches * 100,
            2
        ),
        "%"
    )

    print(
        "ADDRESS BLOCKING RECALL:",
        round(
            address_captured / total_true_matches * 100,
            2
        ),
        "%"
    )

    print(
        "UNION BLOCKING RECALL:",
        round(
            union_captured / total_true_matches * 100,
            2
        ),
        "%"
    )