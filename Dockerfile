FROM python:3.10.6-buster

COPY api /api
COPY grid_intelligence /grid_intelligence
COPY requirements_dev.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
