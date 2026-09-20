FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

WORKDIR /app

# Installed before the source is copied so that editing code does not invalidate
# the (slow) TensorFlow layer on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src
COPY templates ./templates
COPY scripts ./scripts

# Trained weights. If models/ is empty at build time the app still starts and
# says so in the UI; mount or COPY the .keras files to enable predictions.
COPY models ./models

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"

# One worker: two TensorFlow workers would each load a full copy of both models
# into memory, which exceeds the RAM on most free hosting tiers. Threads handle
# concurrent uploads instead.
CMD gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT
