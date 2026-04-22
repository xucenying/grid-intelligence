FROM python:3.10.6-buster

COPY grid_intelligence /grid_intelligence
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn grid_intelligence.api.fast:app --host 0.0.0.0 --port $PORT
