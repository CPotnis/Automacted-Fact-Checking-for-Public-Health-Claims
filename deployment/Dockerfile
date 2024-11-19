FROM python:3.9-slim

# Set the working directory.
WORKDIR /app

# Copy application files.
COPY ../src/serve.py ./
COPY ../src/helper/ /app/helper/
COPY ../requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build -f deployments/Dockerfile -t claim-veracity-api .