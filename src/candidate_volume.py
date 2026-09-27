import pandas as pd
import re
import unicodedata


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "dataset/train"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKC", value)

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

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

    digits = address.str.replace(
        r"\D",
        "",
        regex=True
    )

    keys = pd.DataFrame(index=df.index)

    # Name blocks
    keys["name_prefix8"] = (
        country + "|" + name_compact.str[:8]
    )

    keys["name_prefix4"] = (
        country + "|" + name_compact.str[:4]
    )

    # Address blocks
    keys["address_prefix6"] = (
        country + "|" + address_compact.str[:6]
    )

    keys["address_prefix3"] = (
        country + "|" + address_compact.str[:3]
    )

    # Address number block
    keys["address_digits"] = (
        country + "|" + digits.str[:5]
    )

    return keys


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

s1 = pd.read_csv(
    f"{TRAIN_DIR}/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    f"{TRAIN_DIR}/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    f"{TRAIN_DIR}/train_source3.tsv",
    sep="\t"
)

print("Data loaded.")


# ============================================================
# CREATE KEYS
# ============================================================

print("Creating blocking keys...")

s1_keys = create_keys(s1)
s2_keys = create_keys(s2)
s3_keys = create_keys(s3)


# ============================================================
# COMBINATION DEFINITIONS
# ============================================================

combinations = {

    "name_only": [
        "name_prefix8",
        "name_prefix4"
    ],

    "name_address": [
        "name_prefix8",
        "name_prefix4",
        "address_prefix6",
        "address_prefix3"
    ],

    "name_address_digits": [
        "name_prefix8",
        "name_prefix4",
        "address_prefix6",
        "address_prefix3",
        "address_digits"
    ]
}


# ============================================================
# ANALYZE ONE COMBINATION
# ============================================================

def analyze_combination(
    combination_name,
    keys
):

    print("\n========================================")
    print("Testing:", combination_name)
    print("========================================")

    # --------------------------------------------------------
    # Build block sizes
    # --------------------------------------------------------

    block_sizes = {}

    for key in keys:

        # Combine S2 + S3
        values = pd.concat(
            [
                s2_keys[key],
                s3_keys[key]
            ],
            ignore_index=True
        )

        counts = values.value_counts()

        block_sizes[key] = counts

    # --------------------------------------------------------
    # Calculate candidate count for every S1
    # --------------------------------------------------------

    candidate_counts = []

    for key in keys:

        counts = block_sizes[key]

        # Number of S2/S3 records sharing S1's block
        counts_for_s1 = (
            s1_keys[key]
            .map(counts)
            .fillna(0)
        )

        candidate_counts.append(
            counts_for_s1
        )

    # --------------------------------------------------------
    # UNION candidate estimate
    # --------------------------------------------------------
    #
    # Same candidate can appear in multiple blocks.
    #
    # Therefore summing block counts would over-count.
    #
    # To estimate union size correctly, we build
    # candidate IDs only per S1 block.
    #
    # We process S1 one at a time to control memory.
    # --------------------------------------------------------

    # Build inverted indexes
    indexes = {}

    for key in keys:

        temp = pd.concat(
            [
                s2_keys[[key]].assign(
                    source="S2",
                    row_id=s2["entity_id"].values
                ),
                s3_keys[[key]].assign(
                    source="S3",
                    row_id=s3["entity_id"].values
                )
            ],
            ignore_index=True
        )

        indexes[key] = (
            temp.groupby(key)["row_id"]
            .apply(set)
        )

    # --------------------------------------------------------
    # Count union candidates
    # --------------------------------------------------------

    union_counts = []

    for i in range(len(s1)):

        candidate_set = set()

        for key in keys:

            block = s1_keys.iloc[i][key]

            if block == "":
                continue

            ids = indexes[key].get(block)

            if ids is not None:
                candidate_set.update(ids)

        union_counts.append(
            len(candidate_set)
        )

        # Progress
        if (i + 1) % 100000 == 0:
            print(
                f"Processed S1: {i + 1:,} / {len(s1):,}"
            )

    counts = pd.Series(union_counts)

    result = {
        "combination": combination_name,
        "total_candidate_pairs": int(counts.sum()),
        "average_candidates": counts.mean(),
        "median_candidates": counts.median(),
        "p90_candidates": counts.quantile(0.90),
        "p95_candidates": counts.quantile(0.95),
        "p99_candidates": counts.quantile(0.99),
        "max_candidates": counts.max()
    }

    print("\nResults:")

    print(
        f"Total candidates : {result['total_candidate_pairs']:,}"
    )

    print(
        f"Average          : {result['average_candidates']:.2f}"
    )

    print(
        f"Median           : {result['median_candidates']:.2f}"
    )

    print(
        f"P90              : {result['p90_candidates']:.2f}"
    )

    print(
        f"P95              : {result['p95_candidates']:.2f}"
    )

    print(
        f"P99              : {result['p99_candidates']:.2f}"
    )

    print(
        f"Maximum          : {result['max_candidates']:,}"
    )

    return result


# ============================================================
# RUN ALL COMBINATIONS
# ============================================================

results = []

for combination_name, keys in combinations.items():

    result = analyze_combination(
        combination_name,
        keys
    )

    results.append(result)


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "output/candidate_volume.tsv",
    sep="\t",
    index=False
)

print("\n========================================")
print("FINAL CANDIDATE VOLUME REPORT")
print("========================================")

print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print("output/candidate_volume.tsv")