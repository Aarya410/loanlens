FROM python:3.12-slim

ARG GIT_SHA=dev
ARG BUILD_TIME=unknown
ENV GIT_SHA=$GIT_SHA BUILD_TIME=$BUILD_TIME PYTHONUNBUFFERED=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train the model while building the image, so every deploy ships a fresh model
RUN python train.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
