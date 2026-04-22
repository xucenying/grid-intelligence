include .env
export

install:
	pip install -r requirements.txt

reload:
	pip install -e .

fetch:
	python -c "from grid_intelligence.params import *; from grid_intelligence.data.fetcher import EnergyDataFetcher; from datetime import datetime; f = EnergyDataFetcher(ENTSOE_API_KEY, EIA_API_KEY, DATA_DIR); f.fetch_all('2024-04-21', datetime.now().strftime('%Y-%m-%d'))"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
