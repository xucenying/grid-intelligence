FROM python:3.10.6-buster

ARG BUILD_DATE

COPY api /api
COPY grid_intelligence /grid_intelligence
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
