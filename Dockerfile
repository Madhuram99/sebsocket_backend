# Collections ROI Copilot - Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  postgresql-client \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY copilot_state.py .
COPY copilot_graph.py .
COPY copilot_api.py .

# Create directories
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run the application
CMD ["uvicorn", "copilot_api:app", "--host", "0.0.0.0", "--port", "8080"]
