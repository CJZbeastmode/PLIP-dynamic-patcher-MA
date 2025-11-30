
from downloader.image_downloader import ImageDownloader

out_dir = "/Volumes/Xbox_HD/Data/med_img"
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
    "TCGA-SKCM": 0.744
}

#id = ImageDownloader(proportions=PROPORTIONS, out_dir=out_dir, target_total=600)
#id.run()


down = ImageDownloader(proportions=PROPORTIONS, out_dir=out_dir, target_total=600)
down.sample_cases()          # this regenerates selected_cases exactly the same
down.continue_download()
