# grid_intelligence/params.py
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
EIA_API_KEY    = os.getenv("EIA_API_KEY")

print("ENTSOE KEY LOADED:", ENTSOE_API_KEY)
print("EIA KEY LOADED:", EIA_API_KEY)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "raw_data"

COUNTRY        = "DE_LU"
START_DATE     = "2024-04-21"
