import os, requests, time
from bs4 import BeautifulSoup
from tqdm import tqdm


## Configuration
BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

TCGA_KEYWORDS = {
    "TCGA-COAD": "colon adenocarcinoma OR colorectal adenocarcinoma",
    "TCGA-READ": "rectal adenocarcinoma OR colorectal adenocarcinoma",
    "TCGA-ESCA": "esophageal carcinoma",
    "TCGA-STAD": "stomach adenocarcinoma OR gastric cancer",
    "TCGA-LUAD": "lung adenocarcinoma",
    "TCGA-LUSC": "lung squamous cell carcinoma",
    "TCGA-MESO": "mesothelioma",
    "TCGA-CHOL": "cholangiocarcinoma OR bile duct cancer",
    "TCGA-LIHC": "liver hepatocellular carcinoma",
    "TCGA-PAAD": "pancreatic adenocarcinoma",
    "TCGA-UVM":  "uveal melanoma",
    "TCGA-SKCM": "skin cutaneous melanoma",
}

OUTPUT_DIR = "epmc_tcga_corpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PER_PROJECT = 50


## Loop function
def fetch_for_project(code, keyword_query, out_dir, per_project=50):
    """
    Fetch up to 'per_project' articles from EuropePMC for the given cancer type.
    Saves texts into out_dir/<code>/*.txt
    """
    save_dir = os.path.join(out_dir, code)
    os.makedirs(save_dir, exist_ok=True)

    cursor = "*"
    downloaded = 0
    page_size = 25

    print(f"\n=== Fetching for {code}: {keyword_query} ===")

    while True:
        params = {
            "query": f'({keyword_query}) AND OPEN_ACCESS:Y',
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor
        }

        r = requests.get(BASE_URL, params=params)
        data = r.json()

        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        for hit in tqdm(results, desc=f"{code} batch"):
            pmcid = hit.get("pmcid")
            if not pmcid:
                continue

            # Fetch fulltext XML
            url_xml = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            xml = requests.get(url_xml)
            if xml.status_code != 200:
                continue

            soup = BeautifulSoup(xml.text, "xml")
            snippets = []

            # FIGURE CAPTIONS
            for fig in soup.find_all("fig"):
                cap = fig.find("caption")
                if cap:
                    snippets.append(cap.get_text(" ", strip=True))

            # HISTOPATHOLOGY / RESULTS SECTIONS
            for sec in soup.find_all("sec"):
                title = sec.find("title").get_text().lower() if sec.find("title") else ""
                if any(k in title for k in ["histopath", "microscop", "result"]):
                    snippets.append(sec.get_text(" ", strip=True))

            if not snippets:
                continue

            text = "\n".join(snippets)
            fname = os.path.join(save_dir, f"{pmcid}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(text)

            downloaded += 1
            time.sleep(0.2)

            if downloaded >= per_project:
                return downloaded

        cursor = data.get("nextCursorMark")
        if not cursor:
            break

    return downloaded


## Main loop
for code, query in TCGA_KEYWORDS.items():
    count = fetch_for_project(
        code=code,
        keyword_query=query,
        out_dir=OUTPUT_DIR,
        per_project=PER_PROJECT
    )
    print(f"{code}: downloaded {count} articles")
