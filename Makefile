include .env
export

install:
	pip install -r requirements.txt

reload:
	pip install -e .

.PHONY: fetch-full fetch-delta
fetch-full:
	python notebooks/javier/fetcher.py --mode full --start 2018-01-01

fetch-delta:
	python notebooks/javier/fetcher.py --mode delta

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
