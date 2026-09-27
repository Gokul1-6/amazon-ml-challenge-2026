import pandas as pd
import re
import unicodedata


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = "dataset/train"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKC", value)

    # Unicode-safe
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

def create_keys(df):

    name = df["business_name"].fillna("").map(normalize_text)
    address = df["business_address"].fillna("").map(normalize_text)
    country = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    name_compact = name.str.replace(" ", "", regex=False)
    address_compact = address.str.replace(" ", "", regex=False)

    keys = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    keys["name_prefix8"] = (
        country + "|" + name_compact.str[:8]
    )

    keys["name_prefix4"] = (
        country + "|" + name_compact.str[:4]
    )

    # --------------------------------------------------------
    # ADDRESS BLOCKS
    # --------------------------------------------------------

    keys["address_prefix6"] = (
        country + "|" + address_compact.str[:6]
    )

    keys["address_prefix3"] = (
        country + "|" + address_compact.str[:3]
    )

    # --------------------------------------------------------
    # ADDRESS DIGITS
    # --------------------------------------------------------

    digits = address.str.replace(
        r"\D",
        "",
        regex=True
    )

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

gt = pd.read_csv(
    f"{TRAIN_DIR}/train_ground_truth.tsv",
    sep="\t",
    keep_default_na=False
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
# CREATE LOOKUPS
# ============================================================

print("Creating lookups...")

s1_lookup = s1_keys.copy()
s1_lookup["entity_id"] = s1["entity_id"].values
s1_lookup = s1_lookup.set_index("entity_id")

s2_lookup = s2_keys.copy()
s2_lookup["entity_id"] = s2["entity_id"].values
s2_lookup = s2_lookup.set_index("entity_id")

s3_lookup = s3_keys.copy()
s3_lookup["entity_id"] = s3["entity_id"].values
s3_lookup = s3_lookup.set_index("entity_id")


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

gt = gt[
    ["source1_entity_id", "matched_entity_ids"]
].copy()

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)

gt = gt[
    gt["matched_entity_ids"] != ""
]

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .str.split(",")
)

gt = gt.explode(
    "matched_entity_ids",
    ignore_index=True
)

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .astype(str)
    .str.strip()
)

total_pairs = len(gt)

print(
    f"Total true pairs: {total_pairs:,}"
)


# ============================================================
# CHECK COMBINATIONS
# ============================================================

key_sets = {

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
# EVALUATE
# ============================================================

results = []

chunk_size = 500_000


for combination_name, keys in key_sets.items():

    print("\n========================================")
    print("Testing:", combination_name)
    print("Keys:", ", ".join(keys))
    print("========================================")

    captured_total = 0

    for start in range(
        0,
        total_pairs,
        chunk_size
    ):

        chunk = gt.iloc[
            start:start + chunk_size
        ].copy()

        # --------------------------------------------
        # S1 keys
        # --------------------------------------------

        s1_rows = s1_lookup.loc[
            chunk["source1_entity_id"]
        ]

        # --------------------------------------------
        # Determine S2 / S3
        # --------------------------------------------

        ids = chunk["matched_entity_ids"]

        is_s2 = ids.str.startswith("S2-")
        is_s3 = ids.str.startswith("S3-")

        captured = pd.Series(
            False,
            index=chunk.index
        )

        # --------------------------------------------
        # Compare each blocking key
        # --------------------------------------------

        for key in keys:

            s1_values = s1_rows[key].values

            target_values = pd.Series(
                index=chunk.index,
                dtype="object"
            )

            if is_s2.any():

                target_values.loc[is_s2] = (
                    s2_lookup[key]
                    .reindex(ids[is_s2])
                    .values
                )

            if is_s3.any():

                target_values.loc[is_s3] = (
                    s3_lookup[key]
                    .reindex(ids[is_s3])
                    .values
                )

            # A true pair is captured if ANY
            # blocking key matches.

            captured |= (
                s1_values
                == target_values.values
            )

        captured_total += captured.sum()

    recall = (
        captured_total
        / total_pairs
        * 100
    )

    results.append({
        "combination": combination_name,
        "captured_pairs": int(captured_total),
        "total_true_pairs": total_pairs,
        "recall_percent": recall
    })

    print(
        f"Captured: {captured_total:,}"
    )

    print(
        f"Recall: {recall:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "output/blocking_v3_recall.tsv",
    sep="\t",
    index=False
)

print("\n========================================")
print("BLOCKING V3 FINAL RESULTS")
print("========================================")

print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print("output/blocking_v3_recall.tsv")