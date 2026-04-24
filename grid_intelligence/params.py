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

RENEWABLE = [
    'Biomass', 'Geothermal', 'Hydro Pumped Storage',
    'Hydro Run-of-river and poundage', 'Hydro Water Reservoir',
    'Other renewable', 'Solar', 'Wind Offshore', 'Wind Onshore'
]

NON_RENEWABLE = [
    'Fossil Brown coal/Lignite', 'Fossil Coal-derived gas',
    'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil', 'Waste', 'Other'
]
