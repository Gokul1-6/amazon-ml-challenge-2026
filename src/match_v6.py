import pandas as pd
import re
import unicodedata
from rapidfuzz.fuzz import ratio, token_set_ratio
import os

TEST = "dataset/test"
CANDIDATES = "output/candidate_pairs_v6.tsv"
OUTPUT = "output/matching_results_v6.tsv"

# Conservative thresholds because challenge uses precision-heavy F0.5
NAME_THRESHOLD = 0.82
ADDRESS_THRESHOLD = 0.55
COMBINED_THRESHOLD = 0.70


def norm(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKC", str(x)).lower()
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def similarity(s1_name, s1_addr, s2_name, s2_addr):

    n1 = norm(s1_name)
    n2 = norm(s2_name)

    a1 = norm(s1_addr)
    a2 = norm(s2_addr)

    if not n1 or not n2:
        name_score = 0.0
    else:
        r = ratio(n1, n2) / 100.0
        t = token_set_ratio(n1, n2) / 100.0
        name_score = 0.60 * r + 0.40 * t

    if not a1 or not a2:
        addr_score = 0.0
    else:
        r = ratio(a1, a2) / 100.0
        t = token_set_ratio(a1, a2) / 100.0
        addr_score = 0.60 * r + 0.40 * t

    combined = (
        0.60 * name_score +
        0.40 * addr_score
    )

    return name_score, addr_score, combined


print("Loading test source files...")

s1 = pd.read_csv(
    f"{TEST}/test_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    f"{TEST}/test_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    f"{TEST}/test_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


print("Building candidate lookup...")

# Only fields needed for matching
lookup = {}

for row in s2.itertuples(index=False):

    lookup[row.entity_id] = (
        row.business_name,
        row.business_address,
        row.country
    )

for row in s3.itertuples(index=False):

    lookup[row.entity_id] = (
        row.business_name,
        row.business_address,
        row.country
    )

print("Lookup entries:", len(lookup))


print("Opening candidate file...")

os.makedirs("output", exist_ok=True)

# Read candidate file in chunks.
CHUNK_SIZE = 5000

reader = pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)


with open(OUTPUT, "w", encoding="utf-8") as out:

    out.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    processed = 0

    for chunk in reader:

        for row in chunk.itertuples(index=False):

            s1_id = row.source1_entity_id

            # Get S1 record
            s1_row = s1.loc[
                s1["entity_id"] == s1_id
            ]

            if len(s1_row) == 0:
                out.write(s1_id + "\t\n")
                continue

            s1_data = s1_row.iloc[0]

            s1_name = s1_data["business_name"]
            s1_addr = s1_data["business_address"]
            s1_country = norm(s1_data["country"])

            candidates = row.candidate_entity_ids

            if not candidates:
                out.write(s1_id + "\t\n")
                continue

            matches = []

            for candidate_id in candidates.split(","):

                if candidate_id not in lookup:
                    continue

                c_name, c_addr, c_country = lookup[candidate_id]

                # Country must match
                if s1_country != norm(c_country):
                    continue

                name_score, addr_score, combined = similarity(
                    s1_name,
                    s1_addr,
                    c_name,
                    c_addr
                )

                # Strong exact-name match
                exact_name = (
                    norm(s1_name) != ""
                    and norm(s1_name) == norm(c_name)
                )

                # Strong matching conditions
                accept = False

                if exact_name and addr_score >= 0.30:
                    accept = True

                elif (
                    name_score >= NAME_THRESHOLD
                    and addr_score >= ADDRESS_THRESHOLD
                    and combined >= COMBINED_THRESHOLD
                ):
                    accept = True

                if accept:
                    matches.append(
                        (candidate_id, combined)
                    )

            # Sort strongest matches first
            matches.sort(
                key=lambda x: x[1],
                reverse=True
            )

            matched_ids = [
                x[0]
                for x in matches
            ]

            out.write(
                s1_id
                + "\t"
                + ",".join(matched_ids)
                + "\n"
            )

        processed += len(chunk)

        print(
            f"Processed S1: "
            f"{processed:,} / {len(s1):,}"
        )


print()
print("DONE")
print("Output:", OUTPUT)