FROM python:3.9

WORKDIR /app
COPY requirements.txt /app

RUN pip3 install -r requirements.txt

COPY . /app

ENV PORT=8080
EXPOSE $PORT


CMD uvicorn main:app --host 0.0.0.0 --reload --port ${PORT}