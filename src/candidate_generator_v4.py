import pandas as pd
import re
import unicodedata


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"
OUTPUT_FILE = "output/candidate_pairs.tsv"

BLOCKING_KEYS = [
    "name_prefix8",
    "name_prefix4",
    "address_prefix6",
    "address_prefix3",
    "address_digits",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Unicode-safe normalization
    value = unicodedata.normalize("NFKC", value)

    # Keep Unicode letters and numbers
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Remove repeated spaces
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

def create_keys(df):

    name = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    address = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    country = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Remove spaces
    name_compact = name.str.replace(
        " ",
        "",
        regex=False
    )

    address_compact = address.str.replace(
        " ",
        "",
        regex=False
    )

    # Extract address digits
    digits = address.str.replace(
        r"\D",
        "",
        regex=True
    )

    keys = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    keys["name_prefix8"] = (
        country
        + "|"
        + name_compact.str[:8]
    )

    keys["name_prefix4"] = (
        country
        + "|"
        + name_compact.str[:4]
    )

    # --------------------------------------------------------
    # ADDRESS BLOCKS
    # --------------------------------------------------------

    keys["address_prefix6"] = (
        country
        + "|"
        + address_compact.str[:6]
    )

    keys["address_prefix3"] = (
        country
        + "|"
        + address_compact.str[:3]
    )

    # --------------------------------------------------------
    # ADDRESS DIGITS
    # --------------------------------------------------------

    keys["address_digits"] = (
        country
        + "|"
        + digits.str[:5]
    )

    return keys


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

s1 = pd.read_csv(
    f"{TEST_DIR}/test_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s2 = pd.read_csv(
    f"{TEST_DIR}/test_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s3 = pd.read_csv(
    f"{TEST_DIR}/test_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Loaded:")
print("S1:", f"{len(s1):,}")
print("S2:", f"{len(s2):,}")
print("S3:", f"{len(s3):,}")


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

print("\nCreating blocking keys...")

s1_keys = create_keys(s1)
s2_keys = create_keys(s2)
s3_keys = create_keys(s3)

print("Blocking keys created.")


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_index(keys_df, entity_ids):

    indexes = {}

    for key in BLOCKING_KEYS:

        print(
            f"Building index: {key}"
        )

        temp = pd.DataFrame({
            "block": keys_df[key].values,
            "entity_id": entity_ids.values
        })

        # Empty blocks are useless
        temp = temp[
            temp["block"] != ""
        ]

        # block -> comma-separated entity IDs
        grouped = (
            temp
            .groupby("block")["entity_id"]
            .agg(",".join)
        )

        indexes[key] = grouped

        print(
            f"  Blocks: {len(grouped):,}"
        )

    return indexes


# ============================================================
# BUILD S2 / S3 INDEXES
# ============================================================

print("\nBuilding S2 indexes...")

s2_indexes = build_index(
    s2_keys,
    s2["entity_id"]
)

print("\nBuilding S3 indexes...")

s3_indexes = build_index(
    s3_keys,
    s3["entity_id"]
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating candidate pairs...")

BATCH_SIZE = 25_000

output_parts = []

total_s1 = len(s1)


for start in range(
    0,
    total_s1,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_s1
    )

    batch_s1 = s1.iloc[
        start:end
    ]

    batch_keys = s1_keys.iloc[
        start:end
    ]

    # --------------------------------------------------------
    # Get candidates from each blocking key
    # --------------------------------------------------------

    candidate_columns = []

    for key in BLOCKING_KEYS:

        # S2 candidates
        s2_matches = (
            batch_keys[key]
            .map(s2_indexes[key])
            .fillna("")
        )

        # S3 candidates
        s3_matches = (
            batch_keys[key]
            .map(s3_indexes[key])
            .fillna("")
        )

        combined = (
            s2_matches
            + ","
            + s3_matches
        )

        candidate_columns.append(
            combined
        )

    # --------------------------------------------------------
    # Union candidates for each S1
    # --------------------------------------------------------

    rows = []

    for row_index in range(
        len(batch_s1)
    ):

        candidate_set = set()

        for column in candidate_columns:

            value = column.iloc[
                row_index
            ]

            if not value:
                continue

            for entity_id in value.split(","):

                if entity_id:
                    candidate_set.add(
                        entity_id
                    )

        rows.append({
            "source1_entity_id":
                batch_s1.iloc[
                    row_index
                ]["entity_id"],

            "candidate_entity_ids":
                ",".join(
                    sorted(candidate_set)
                )
        })

    output_parts.append(
        pd.DataFrame(rows)
    )

    print(
        f"Processed S1: "
        f"{end:,} / {total_s1:,}"
    )


# ============================================================
# COMBINE ALL BATCHES
# ============================================================

print("\nCombining candidate batches...")

candidate_df = pd.concat(
    output_parts,
    ignore_index=True
)


# ============================================================
# SAVE CANDIDATE FILE
# ============================================================

print("\nSaving candidate_pairs.tsv...")

candidate_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

candidate_counts = (
    candidate_df[
        "candidate_entity_ids"
    ]
    .fillna("")
    .map(
        lambda x:
        0 if not x
        else len(x.split(","))
    )
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("CANDIDATE GENERATION COMPLETE")
print("========================================")

print(
    "S1 records:",
    f"{len(candidate_df):,}"
)

print(
    "Total candidate pairs:",
    f"{candidate_counts.sum():,}"
)

print(
    "Average candidates/S1:",
    f"{candidate_counts.mean():.2f}"
)

print(
    "Median candidates/S1:",
    f"{candidate_counts.median():.2f}"
)

print(
    "P95 candidates/S1:",
    f"{candidate_counts.quantile(0.95):.2f}"
)

print(
    "Maximum candidates/S1:",
    f"{candidate_counts.max():,}"
)

print()
print("Output file:")
print(OUTPUT_FILE)