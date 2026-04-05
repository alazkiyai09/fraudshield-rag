FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/sample/ ./data/sample/
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app

EXPOSE 8080
USER appuser
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
