"""
Central config for MAPCOS.
Update the two DATA_DIR paths once you've added the datasets in Kaggle
and can see their real mounted paths under /kaggle/input/.
"""

IMG_SIZE = 224
BATCH_SIZE = 32

DATA_DIR_VISION = "/kaggle/input/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images/data"
DATA_DIR_TABULAR = "/kaggle/input/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos"



PCOM_THRESHOLD = 20            # Rotterdam PCOM: >=20 follicles per ovary
ROTTERDAM_CRITERIA_COUNT = 2   # 2-of-3 rule

# Follicle Counting Agent — Hough Circle params
HOUGH_PARAMS = {
    "dp": 1.2,
    "min_dist": 15,
    "param1": 50,
    "param2": 25,
    "min_radius": 3,
    "max_radius": 25,
}

# Follicle Counting Agent — watershed segmentation params
FOLLICLE_PARAMS = {
    "crop_fraction": 0.88,          # keeps top 88% of the image, drops the flat annotation bar
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,
    "min_peak_distance": 9,         # merges near-duplicate detections
    "min_area": 15,                 # loosened from 20 — was rejecting real small follicles
    "max_area": 500,
    "min_circularity": 0.4,         # loosened from 0.45 — same reason
    "border_margin": 3,             # drops crop-edge artifacts
}
# Agreement / Grounding Agent
GROUNDING_PARAMS = {
    "heatmap_threshold": 0.5,      # Grad-CAM value above which a pixel counts as "CNN attention"
    "iou_agreement_threshold": 0.15,  # IoU rarely gets high here — Grad-CAM is diffuse, follicles are small circles
}
# Tabular data — Lab & Symptoms agents
TABULAR_XLSX_PATH = f"{DATA_DIR_TABULAR}/PCOS_data_without_infertility.xlsx"
TABULAR_SHEET_NAME = "Full_new"
TARGET_COLUMN = "PCOS (Y/N)"

LAB_MODEL_PATH = "lab_agent_xgb.json"
SYMPTOMS_MODEL_PATH = "symptoms_agent_xgb.json"

# Real column names in PCOS_data_without_infertility.xlsx, sheet "Full_new"
LAB_RAW_COLUMNS = {
    "fsh": "FSH(mIU/mL)",
    "lh": "LH(mIU/mL)",
    "amh": "AMH(ng/mL)",
}
SYMPTOMS_RAW_COLUMNS = {
    "bmi": "BMI",
    "cycle_code": "Cycle(R/I)",        # 2 = Regular; anything else (4, or a stray 5) = Irregular
    "hair_growth": "hair growth(Y/N)",
    "weight_gain": "Weight gain(Y/N)",
    "skin_darkening": "Skin darkening (Y/N)",
    "hair_loss": "Hair loss(Y/N)",
    "pimples": "Pimples(Y/N)",
}
CNN_MODEL_PATH = "cnn_agent_best.h5"
LAST_CONV_LAYER = "backbone"          # changed from "efficientnetb0" — compare_cnns.py names it this way for all backbones

LAB_MODEL_PATH = "lab_agent_best.joblib"       # changed from .json — now a joblib-saved sklearn-style model
SYMPTOMS_MODEL_PATH = "symptoms_agent_best.joblib"  # changed from .json
