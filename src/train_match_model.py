import pandas as pd
import numpy as np
import re
import unicodedata
from rapidfuzz.fuzz import ratio, token_set_ratio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import joblib
import os

TRAIN = "dataset/train"

# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def norm(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKC", str(x)).lower()
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------

def features(a, b):

    # Source-1 record is a pandas Series.
    # Candidate record may be a dictionary.
    if isinstance(b, dict):

        name_a = a["business_name"]
        name_b = b["business_name"]

        addr_a = a["business_address"]
        addr_b = b["business_address"]

        country_a = a["country"]
        country_b = b["country"]

    else:

        name_a = a["business_name"]
        name_b = b.business_name

        addr_a = a["business_address"]
        addr_b = b.business_address

        country_a = a["country"]
        country_b = b.country

    n1 = norm(name_a)
    n2 = norm(name_b)

    a1 = norm(addr_a)
    a2 = norm(addr_b)

    c1 = norm(country_a)
    c2 = norm(country_b)

    name_ratio = ratio(n1, n2) / 100.0
    name_token = token_set_ratio(n1, n2) / 100.0

    address_ratio = ratio(a1, a2) / 100.0
    address_token = token_set_ratio(a1, a2) / 100.0

    country_match = int(c1 == c2)

    exact_name = int(
        n1 == n2 and n1 != ""
    )

    exact_address = int(
        a1 == a2 and a1 != ""
    )

    return [
        name_ratio,
        name_token,
        address_ratio,
        address_token,
        country_match,
        exact_name,
        exact_address,
    ]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading training data...", flush=True)

s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1: {len(s1):,}", flush=True)
print(f"S2: {len(s2):,}", flush=True)
print(f"S3: {len(s3):,}", flush=True)
print(f"Ground truth: {len(gt):,}", flush=True)


# ---------------------------------------------------------
# FAST LOOKUPS
# ---------------------------------------------------------

print("Building source lookups...", flush=True)

s1_map = s1.set_index("entity_id").to_dict("index")
s2_map = s2.set_index("entity_id").to_dict("index")
s3_map = s3.set_index("entity_id").to_dict("index")


# ---------------------------------------------------------
# CREATE TRAINING PAIRS
# ---------------------------------------------------------

print("Creating training pairs...", flush=True)

X = []
y = []

rng = np.random.default_rng(42)

# Country-indexed pools for negative sampling
country_s2 = {}

for country, group in s2.groupby("country"):

    country_s2[country] = group


# Process a manageable number of S1 records
MAX_S1 = 100_000

processed = 0

for _, row in gt.iterrows():

    if processed >= MAX_S1:
        break

    sid = row["source1_entity_id"]

    a_dict = s1_map.get(sid)

    if a_dict is None:
        continue

    positives = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    positive_set = set(positives)

    # -----------------------------------------------------
    # POSITIVE PAIRS
    # -----------------------------------------------------

    for eid in positives:

        b_dict = s2_map.get(eid)

        if b_dict is None:
            b_dict = s3_map.get(eid)

        if b_dict is None:
            continue

        X.append(features(a_dict, b_dict))
        y.append(1)

    # -----------------------------------------------------
    # HARD NEGATIVES
    # -----------------------------------------------------

    country = a_dict["country"]

    pool = country_s2.get(country)

    if pool is not None and len(pool) > 0:

        sample_size = min(5, len(pool))

        sample = pool.sample(
            n=sample_size,
            random_state=int(
                rng.integers(1, 1_000_000)
            )
        )

        for _, neg in sample.iterrows():

            eid = neg["entity_id"]

            if eid in positive_set:
                continue

            X.append(
                features(
                    a_dict,
                    neg
                )
            )

            y.append(0)

    processed += 1

    if processed % 10_000 == 0:

        print(
            f"Processed training S1: "
            f"{processed:,} / {MAX_S1:,}",
            flush=True
        )


# ---------------------------------------------------------
# NUMPY DATA
# ---------------------------------------------------------

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int8
)

print()
print("==============================")
print("TRAINING DATA")
print("==============================")

print(
    "Pairs:",
    f"{len(y):,}"
)

print(
    "Positive:",
    int(y.sum())
)

print(
    "Negative:",
    int((y == 0).sum())
)

print(
    "Features:",
    X.shape[1]
)


# ---------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------

print()
print("Creating validation split...", flush=True)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Train:",
    len(y_train)
)

print(
    "Validation:",
    len(y_val)
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

print()
print("Training HistGradientBoosting model...", flush=True)

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model training complete.", flush=True)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

print()
print("Finding best F0.5 threshold...", flush=True)

prob = model.predict_proba(
    X_val
)[:, 1]

best_score = -1.0
best_threshold = 0.50

for threshold in np.arange(
    0.30,
    0.96,
    0.01
):

    pred = (
        prob >= threshold
    ).astype(np.int8)

    score = fbeta_score(
        y_val,
        pred,
        beta=0.5,
        zero_division=0
    )

    if score > best_score:

        best_score = score
        best_threshold = float(threshold)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("========================================")
print("MODEL VALIDATION RESULT")
print("========================================")

print(
    "Validation F0.5:",
    round(best_score, 6)
)

print(
    "Best threshold:",
    round(best_threshold, 2)
)

print("========================================")


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model_path = (
    "models/match_model.pkl"
)

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
    },
    model_path
)

print()
print(
    "Saved model:",
    model_path
)

print("DONE")