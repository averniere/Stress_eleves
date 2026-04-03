import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for folder in [LOGS_DIR, MODELS_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 8
C_GRID_LASSO = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

#DEFAULT_SUBSET_VARS = [
    #'reussite_academique', 'qualite_sommeil', 'relation_prof_etudiant', 
    #'besoins_elementaires', 'estime_de_soi', 'problem_respiratoire'
#]

DATA_URL = os.getenv("DATA_PATH", "s3://mon-bucket/stress.csv")
LOG_FORMAT = "{asctime} - {levelname} - {message}"