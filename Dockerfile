FROM python:3.9
ENV PYTHONUNBUFFERED True

WORKDIR /adoptify

COPY ./requirements.txt /adoptify/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /adoptify/requirements.txt

COPY . /adoptify/

CMD uvicorn main:app --port=9000 --host=0.0.0.0