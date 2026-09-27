import pandas as pd
import re
import unicodedata


# ============================================================
# PATHS
# ============================================================

TEST_DIR = "dataset/test"
OUTPUT_FILE = "output/matching_results.tsv"


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Unicode-safe normalization
    value = unicodedata.normalize("NFKC", value)

    # Remove punctuation
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

s1 = pd.read_csv(
    f"{TEST_DIR}/test_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    f"{TEST_DIR}/test_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    f"{TEST_DIR}/test_source3.tsv",
    sep="\t"
)

print("Loaded:")
print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ============================================================
# CREATE NORMALIZED NAME
# ============================================================

print("Normalizing names...")

s1["name_norm"] = (
    s1["business_name"]
    .map(normalize_text)
)

s2["name_norm"] = (
    s2["business_name"]
    .map(normalize_text)
)

s3["name_norm"] = (
    s3["business_name"]
    .map(normalize_text)
)


# ============================================================
# CREATE COUNTRY
# ============================================================

s1["country_norm"] = (
    s1["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

s2["country_norm"] = (
    s2["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

s3["country_norm"] = (
    s3["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# BUILD LOOKUP
# ============================================================

print("Building lookup...")

s2_lookup = {}

for row in s2.itertuples(index=False):

    if not row.name_norm:
        continue

    key = (
        row.country_norm,
        row.name_norm
    )

    s2_lookup.setdefault(
        key,
        []
    ).append(row.entity_id)


s3_lookup = {}

for row in s3.itertuples(index=False):

    if not row.name_norm:
        continue

    key = (
        row.country_norm,
        row.name_norm
    )

    s3_lookup.setdefault(
        key,
        []
    ).append(row.entity_id)


# ============================================================
# GENERATE MATCHING RESULTS
# ============================================================

print("Generating matching_results.tsv...")

results = []

for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    key = (
        row.country_norm,
        row.name_norm
    )

    matches = []

    # S2 matches
    matches.extend(
        s2_lookup.get(key, [])
    )

    # S3 matches
    matches.extend(
        s3_lookup.get(key, [])
    )

    # Remove duplicates
    matches = list(
        dict.fromkeys(matches)
    )

    results.append({
        "source1_entity_id": row.entity_id,
        "matched_entity_ids": ",".join(matches)
    })

    if i % 100000 == 0:
        print(
            f"Processed S1: {i:,} / {len(s1):,}"
        )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print()
print("========================================")
print("BASELINE SUBMISSION CREATED")
print("========================================")

print(
    "Rows:",
    len(result_df)
)

print(
    "File:",
    OUTPUT_FILE
)

print()
print(result_df.head(10).to_string(index=False))