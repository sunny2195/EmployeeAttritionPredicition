FROM python:3.12-slim-bullseye

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME


COPY requirements.txt .
COPY setup.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install .

COPY EAP/ EAP/
COPY app.py .
COPY templates/ templates/
COPY artifacts/ artifacts/
COPY schema.yaml .


EXPOSE 5000


CMD ["python3", "app.py"]