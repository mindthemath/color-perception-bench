FROM python:3.10-slim

WORKDIR /app

# Ensure output is not buffered so we can see logs
ENV PYTHONUNBUFFERED=1

# Install git (needed for some pip installs or HF)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set HF_HOME to a local directory and ensure it exists
ENV HF_HOME=/app/model_cache
RUN mkdir -p $HF_HOME

# Pre-download the model into the image
COPY download_model.py .
# RUN python download_model.py && rm download_model.py

COPY app.py .

# Expose port 8002
EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
