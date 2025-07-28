FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .


RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && pip install --no-cache-dir -r requirements.txt && apt-get purge -y build-essential gcc && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
