import requests
import random
import os
from collections import defaultdict


class ImageDownloader:
    GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
    GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
    FIELDS = "file_id,file_name,cases.submitter_id"

    def __init__(self, proportions, target_total=500, out_dir="tcga_wsis"):
        """
        proportions: dict { "TCGA-COAD": proportion_float, ... }
        target_total: desired total number of WSIs
        out_dir: output directory
        """
        self.proportions = proportions
        self.target_total = target_total
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

        self.target_counts = self._compute_target_counts()
        self.selected_cases = {}  # case_id → (file_id, file_name, project)

    def _compute_target_counts(self):
        sum_prop = sum(self.proportions.values())
        return {
            proj: int((p / sum_prop) * self.target_total)
            for proj, p in self.proportions.items()
        }

    def print_target_counts(self):
        print("Target samples per project:")
        for k, v in self.target_counts.items():
            print(f"{k}: {v}")

    # Quering
    def query_project(self, project_id):
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                    {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
                    {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
                ]
            },
            "fields": self.FIELDS,
            "format": "JSON",
            "size": 5000
        }

        r = requests.post(self.GDC_FILES_ENDPOINT, json=payload)
        r.raise_for_status()
        data = r.json()["data"]["hits"]

        entries = []
        for item in data:
            case_id = item["cases"][0]["submitter_id"]
            file_id = item["file_id"]
            file_name = item["file_name"]
            entries.append((case_id, file_id, file_name))
        return entries

    # Sampling
    def sample_cases(self):
        selected = {}

        for project, n_target in self.target_counts.items():
            print(f"\nQuerying project: {project}")
            entries = self.query_project(project)

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
                selected[cid] = (file_id, file_name, project)

        # Adjust if more than target_total
        if len(selected) > self.target_total:
            excess = len(selected) - self.target_total
            print(f"\nTrimming {excess} cases to reach EXACT {self.target_total}")
            keys_to_trim = list(selected.keys())[:excess]
            for k in keys_to_trim:
                del selected[k]

        self.selected_cases = selected
        print("\nFinal selected cases:", len(self.selected_cases))

    # Shuffling
    def shuffled_case_list(self):
        """
        Returns a *shuffled list* of all selected cases.
        Format of each item:
        (case_id, file_id, file_name, project)
        """
        items = [
            (cid, *self.selected_cases[cid])
            for cid in self.selected_cases
        ]
        random.shuffle(items)
        return items
    
    # Download
    def download_one(self, cid, file_id):
        """
        Download a single WSI to a temporary file and return its path.
        """
        url = f"{self.GDC_DATA_ENDPOINT}/{file_id}"
        out_path = os.path.join(self.out_dir, f"{cid}.svs")

        print(f"Downloading {cid} → {out_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return out_path

    def download_all(self):
        print("\nStarting downloads...")
        for cid, (fid, fname, proj) in self.selected_cases.items():
            out_path = os.path.join(self.out_dir, f"{cid}.svs")
            
            # Skip if already downloaded
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"Skipping (already exists): {proj} {cid}")
                continue

            print(f"Downloading {proj}: {cid} → {out_path}")
            url = f"{self.GDC_DATA_ENDPOINT}/{fid}"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

        print("\nDONE. All slides stored in:", self.out_dir)


    # Iterator
    def __iter__(self):
        """
        Iterator over selected TCGA cases.
        Yields tuples: (case_id, file_id, file_name, project)
        """
        for cid, (fid, fname, proj) in self.selected_cases.items():
            yield cid, fid, fname, proj

    def missing_cases(self):
        missing = []
        for cid, (fid, fname, proj) in self.selected_cases.items():
            out_path = os.path.join(self.out_dir, f"{cid}.svs")
            if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                missing.append((cid, fid, fname, proj))
        return missing

    def continue_download(self):
        todo = self.missing_cases()
        print(f"\nMissing files: {len(todo)}")

        for cid, fid, fname, proj in todo:
            out_path = os.path.join(self.out_dir, f"{cid}.svs")
            url = f"{self.GDC_DATA_ENDPOINT}/{fid}"

            print(f"Continuing download: {proj} {cid} → {out_path}")

            try:
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

            except Exception as e:
                print(f"  ERROR downloading {cid} ({proj}): {e}")

                # Delete partial file if it exists
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                        print("  Partial file removed.")
                    except Exception as del_err:
                        print(f"  Could not remove partial file: {del_err}")

                print("  Skipping.\n")
                continue

        print("\nCONTINUE DONE.")


    # ---------------------------------------------------------
    # FULL PIPELINE
    # ---------------------------------------------------------
    def run(self):
        self.print_target_counts()
        self.sample_cases()
        self.download_all()
