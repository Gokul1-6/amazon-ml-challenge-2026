import pandas as pd
import re
import unicodedata
from collections import defaultdict
import os

BASE = "dataset/test"
OUT = "output/candidate_pairs_v6.tsv"

# Maximum candidates allowed for one block
MAX_BLOCK = 100

def norm(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x)).lower()
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def keys(row):
    name = norm(row.business_name)
    addr = norm(row.business_address)
    country = norm(row.country)

    compact_name = name.replace(" ", "")
    digits = re.sub(r"\D", "", addr)

    result = []

    if compact_name:
        result.append(("N8", country + "|" + compact_name[:8]))

    if len(compact_name) >= 4:
        result.append(("N4", country + "|" + compact_name[:4]))

    if len(digits) >= 4:
        result.append(("D6", country + "|" + digits[:6]))

    return result


print("Loading test data...")

s1 = pd.read_csv(
    f"{BASE}/test_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    f"{BASE}/test_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    f"{BASE}/test_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


def build_index(df):
    index = defaultdict(list)

    for row in df.itertuples(index=False):
        for key in keys(row):

            # Keep only selective blocks
            if len(index[key]) < MAX_BLOCK:
                index[key].append(row.entity_id)

    return index


print("Building S2 index...")
idx2 = build_index(s2)

print("Building S3 index...")
idx3 = build_index(s3)

print("S2 blocks:", len(idx2))
print("S3 blocks:", len(idx3))

os.makedirs("output", exist_ok=True)

print("Generating candidates...")

with open(OUT, "w", encoding="utf-8") as f:

    f.write("source1_entity_id\tcandidate_entity_ids\n")

    for i, row in enumerate(s1.itertuples(index=False), 1):

        candidates = set()

        for key in keys(row):

            candidates.update(idx2.get(key, []))
            candidates.update(idx3.get(key, []))

        f.write(
            row.entity_id
            + "\t"
            + ",".join(candidates)
            + "\n"
        )

        if i % 10000 == 0:
            print(
                f"Processed S1: {i:,} / {len(s1):,}"
            )

print("DONE")
print("Output:", OUT)