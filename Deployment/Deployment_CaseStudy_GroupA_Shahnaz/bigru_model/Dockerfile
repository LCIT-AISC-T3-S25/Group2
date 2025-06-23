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
# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port from config.yaml (default 5000)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
