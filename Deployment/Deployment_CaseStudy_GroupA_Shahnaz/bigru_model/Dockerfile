FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 5003

CMD ["python", "app.py"]
