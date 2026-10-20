FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Bind all interfaces inside the container so a published port works. Remote exposure is
# bounded by the host publish mapping (loopback-only in docker-compose.yml).
ENV NEUROCORE_HOST=0.0.0.0

# Run the application
CMD ["python", "main.py"]
