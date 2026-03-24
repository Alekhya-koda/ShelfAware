# Use an official Python base image
FROM python:3.11-slim

# Create a working directory inside the container
WORKDIR /shelfaware_app

# Copy your requirements file
COPY requirements.txt /shelfaware_app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /shelfaware_app

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]