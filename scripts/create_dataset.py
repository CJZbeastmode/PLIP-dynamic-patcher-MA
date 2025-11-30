import requests
import random
import os
from collections import defaultdict

# ---------------------------------------------------------
# 1. TCGA PROJECT PROPORTIONS (computed from your numbers)
# ---------------------------------------------------------

PROPORTIONS = {
    "TCGA-COAD": 0.308,
    "TCGA-READ": 0.262,
    "TCGA-ESCA": 0.583,
    "TCGA-STAD": 0.556,
    "TCGA-LUAD": 0.391,
    "TCGA-LUSC": 0.378,
    "TCGA-MESO": 0.623,
    "TCGA-CHOL": 0.864,
    "TCGA-LIHC": 0.703,
    "TCGA-PAAD": 0.648,
    "TCGA-UVM":  0.647,
    "TCGA-SKCM": 0.744,
}

TARGET_TOTAL = 500
sum_prop = sum(PROPORTIONS.values())

TARGET_COUNTS = {
    proj: int((p / sum_prop) * TARGET_TOTAL)
    for proj, p in PROPORTIONS.items()
}

print("Target samples per project:")
for k, v in TARGET_COUNTS.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------
# 2. Query GDC for all WSIs in a project
# ---------------------------------------------------------

FIELDS = "file_id,file_name,cases.submitter_id"

def query_project(project_id):
    """Return list of (case_id, file_id, file_name) for WSIs."""
    url = "https://api.gdc.cancer.gov/files"
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=",  "content": { "field": "cases.project.project_id", "value": project_id }},
                {"op": "=",  "content": { "field": "data_type", "value": "Slide Image" }},
                {"op": "in", "content": { "field": "data_format", "value": ["SVS"] }},
            ]
        },
        "fields": FIELDS,
        "format": "JSON",
        "size": 5000
    }

    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()["data"]["hits"]

    entries = []
    for item in data:
        case_id = item["cases"][0]["submitter_id"]
        file_id = item["file_id"]
        file_name = item["file_name"]
        entries.append((case_id, file_id, file_name))

    return entries

# ---------------------------------------------------------
# 3. Sample unique cases according to target count
# ---------------------------------------------------------

selected_cases = {}  # case_id -> (file_id, file_name, project)
used_cases = set()

for project, n_target in TARGET_COUNTS.items():
    print(f"\nQuerying project {project} ...")
    entries = query_project(project)

    # Deduplicate by case ID
    by_case = {}
    for case_id, file_id, file_name in entries:
        if case_id not in by_case:
            by_case[case_id] = (file_id, file_name)

    case_ids = list(by_case.keys())
    random.shuffle(case_ids)

    take = min(len(case_ids), n_target)
    case_ids = case_ids[:take]

    for cid in case_ids:
        file_id, file_name = by_case[cid]
        selected_cases[cid] = (file_id, file_name, project)
        used_cases.add(cid)

print("\nTotal selected cases before trimming:", len(selected_cases))

# Trim to EXACT TARGET_TOTAL (500)
if len(selected_cases) > TARGET_TOTAL:
    trim = len(selected_cases) - TARGET_TOTAL
    print(f"Trimming {trim} extra cases to reach {TARGET_TOTAL}")
    excess = list(selected_cases.keys())[:trim]
    for cid in excess:
        del selected_cases[cid]

print("Final count:", len(selected_cases))

# ---------------------------------------------------------
# 4. Download WSIs
# ---------------------------------------------------------

os.makedirs("tcga_wsis", exist_ok=True)

for cid, (fid, fname, proj) in selected_cases.items():
    url = f"https://api.gdc.cancer.gov/data/{fid}"
    out_path = f"tcga_wsis/{cid}.svs"

    print(f"Downloading {proj} / {cid} -> {out_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print("\nDONE. All slides stored in ./tcga_wsis/")
