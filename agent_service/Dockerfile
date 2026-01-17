FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode dan folder static (HTML)
COPY main.py .
COPY static ./static

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]