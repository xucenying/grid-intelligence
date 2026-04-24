# grid_intelligence/params.py
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR / "raw_data")

COUNTRY            = "DE_LU"
START_DATE         = "2018-10-01"
DELTA_OVERLAP_DAYS = 7
FORECAST_DAYS      = 3

# Environment
ENV = os.getenv("ENV", "development")

# BigQuery
GCP_PROJECT  = os.getenv("GCP_PROJECT", "grid-intelligence-2026")
BQ_DATASET   = "grid_intelligence"
BQ_TABLE     = "consolidated"
BQ_TABLE_ID  = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

RENEWABLE = [
    'Biomass', 'Geothermal', 'Hydro Pumped Storage',
    'Hydro Run-of-river and poundage', 'Hydro Water Reservoir',
    'Other renewable', 'Solar', 'Wind Offshore', 'Wind Onshore'
]

NON_RENEWABLE = [
    'Fossil Brown coal/Lignite', 'Fossil Coal-derived gas',
    'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil', 'Waste', 'Other'
]
