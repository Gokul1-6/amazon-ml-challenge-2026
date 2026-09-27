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

    # Unicode-safe normalization
    value = unicodedata.normalize("NFKC", value)

    # Keep Unicode letters/numbers
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# CREATE V2 BLOCKING KEYS
# ============================================================

def create_keys(df):

    name = df["business_name"].fillna("").map(normalize_text)
    address = df["business_address"].fillna("").map(normalize_text)
    country = df["country"].fillna("").astype(str).str.lower().str.strip()

    # Remove spaces for character-based keys
    name_compact = name.str.replace(" ", "", regex=False)
    address_compact = address.str.replace(" ", "", regex=False)

    keys = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # NAME BLOCKS
    # --------------------------------------------------------

    # Existing-style prefix
    keys["name_prefix8"] = (
        country + "|" + name_compact.str[:8]
    )

    # Shorter prefix helps with spelling variations
    keys["name_prefix4"] = (
        country + "|" + name_compact.str[:4]
    )

    # Last characters help with word-order / suffix variations
    keys["name_suffix4"] = (
        country + "|" + name_compact.str[-4:]
    )

    # First + last characters together
    keys["name_edge"] = (
        country
        + "|"
        + name_compact.str[:4]
        + "|"
        + name_compact.str[-4:]
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

    keys["address_suffix3"] = (
        country + "|" + address_compact.str[-3:]
    )

    keys["address_edge"] = (
        country
        + "|"
        + address_compact.str[:3]
        + "|"
        + address_compact.str[-3:]
    )

    # --------------------------------------------------------
    # ADDRESS NUMBER BLOCK
    # --------------------------------------------------------

    digits = address.str.replace(r"\D", "", regex=True)

    keys["address_digits"] = (
        country + "|" + digits.str[:5]
    )

    return keys


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

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

print("Loaded.")


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

print("Creating S1 keys...")
s1_keys = create_keys(s1)

print("Creating S2 keys...")
s2_keys = create_keys(s2)

print("Creating S3 keys...")
s3_keys = create_keys(s3)


# ============================================================
# ENTITY -> KEY LOOKUP
# ============================================================

print("Creating lookup tables...")

s2_lookup = s2_keys.copy()
s2_lookup["entity_id"] = s2["entity_id"].values

s3_lookup = s3_keys.copy()
s3_lookup["entity_id"] = s3["entity_id"].values

s2_lookup = s2_lookup.set_index("entity_id")
s3_lookup = s3_lookup.set_index("entity_id")


# ============================================================
# GROUND TRUTH PAIRS
# ============================================================

print("Preparing ground truth...")

gt_pairs = gt[
    ["source1_entity_id", "matched_entity_ids"]
].copy()

gt_pairs["matched_entity_ids"] = gt_pairs[
    "matched_entity_ids"
].fillna("")

gt_pairs = gt_pairs[
    gt_pairs["matched_entity_ids"] != ""
]

# Convert comma-separated matches into individual rows
gt_pairs["matched_entity_ids"] = gt_pairs[
    "matched_entity_ids"
].str.split(",")

gt_pairs = gt_pairs.explode(
    "matched_entity_ids",
    ignore_index=True
)

gt_pairs["matched_entity_ids"] = (
    gt_pairs["matched_entity_ids"]
    .astype(str)
    .str.strip()
)

print("Total true pairs:", len(gt_pairs))


# ============================================================
# CHECK EACH BLOCKING KEY
# ============================================================

key_columns = list(s1_keys.columns)

results = []


for key in key_columns:

    print("\nChecking:", key)

    # S1 key
    left = s1[
        ["entity_id"]
    ].copy()

    left["block_key"] = s1_keys[key].values

    left = left.set_index("entity_id")

    # S2 key lookup
    s2_map = s2_lookup[key]

    # S3 key lookup
    s3_map = s3_lookup[key]

    captured = 0
    total = len(gt_pairs)

    # Process GT in chunks
    chunk_size = 500_000

    for start in range(0, total, chunk_size):

        chunk = gt_pairs.iloc[
            start:start + chunk_size
        ].copy()

        # S1 blocking key
        s1_key_values = left.loc[
            chunk["source1_entity_id"],
            "block_key"
        ].values

        # Determine whether match belongs to S2 or S3
        ids = chunk["matched_entity_ids"]

        is_s2 = ids.str.startswith("S2-")

        target_keys = pd.Series(
            index=chunk.index,
            dtype="object"
        )

        # S2 matches
        if is_s2.any():
            target_keys.loc[is_s2] = s2_map.reindex(
                ids[is_s2]
            ).values

        # S3 matches
        is_s3 = ids.str.startswith("S3-")

        if is_s3.any():
            target_keys.loc[is_s3] = s3_map.reindex(
                ids[is_s3]
            ).values

        target_key_values = target_keys.values

        captured += (
            s1_key_values == target_key_values
        ).sum()

    recall = captured / total * 100

    results.append({
        "blocking_key": key,
        "captured_pairs": captured,
        "total_true_pairs": total,
        "recall_percent": recall
    })

    print(
        f"Captured: {captured:,} / {total:,}"
    )

    print(
        f"Recall: {recall:.2f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "recall_percent",
    ascending=False
)

results_df.to_csv(
    "output/blocking_v2_recall.tsv",
    sep="\t",
    index=False
)

print("\n========================================")
print("BLOCKING V2 RESULTS")
print("========================================")

print(results_df.to_string(index=False))

print("\nSaved:")
print("output/blocking_v2_recall.tsv")