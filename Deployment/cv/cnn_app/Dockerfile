# Use a lightweight base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all local project files to container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI app on container start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
