import pandas as pd
import re
import unicodedata
from collections import defaultdict
import os

BASE = "dataset/test"
OUT = "output/candidate_pairs_v5.tsv"

MAX_BLOCK = 200

def norm(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x)).lower()
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def make_keys(row):
    name = norm(row.business_name)
    addr = norm(row.business_address)
    country = norm(row.country)

    digits = re.sub(r"\D", "", addr)

    return [
        ("N8", country + "|" + name.replace(" ", "")[:8]),
        ("N4", country + "|" + name.replace(" ", "")[:4]),
        ("AD", country + "|" + digits[:8]),
    ]

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

print("Building capped indexes...")

indexes = {
    "S2": defaultdict(list),
    "S3": defaultdict(list)
}

def build_index(df, source):
    idx = indexes[source]

    for row in df.itertuples(index=False):
        for key in make_keys(row):
            if len(idx[key]) < MAX_BLOCK:
                idx[key].append(row.entity_id)

    print(source, "blocks:", len(idx))

build_index(s2, "S2")
build_index(s3, "S3")

print("Generating candidates...")

os.makedirs("output", exist_ok=True)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")

    for i, row in enumerate(s1.itertuples(index=False), 1):

        candidates = set()

        for key in make_keys(row):

            for eid in indexes["S2"].get(key, []):
                candidates.add(eid)

            for eid in indexes["S3"].get(key, []):
                candidates.add(eid)

        f.write(
            row.entity_id + "\t" + ",".join(candidates) + "\n"
        )

        if i % 10000 == 0:
            print(f"Processed {i:,}/{len(s1):,}")

print("DONE")
print("Output:", OUT)