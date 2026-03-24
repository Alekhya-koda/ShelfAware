# Stage 1: Build the frontend application
FROM node:20-alpine AS frontend-builder
WORKDIR /app

# Copy package files and install dependencies
COPY ref_frontend/package*.json ./
RUN npm install

# Set build-time environment variable for the frontend
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Copy the rest of the frontend source code and build the application
COPY ref_frontend/ .
RUN npm run build

# Stage 2: Build the final application image
FROM python:3.11-slim
WORKDIR /app

# Install backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy backend application code
COPY . ./app

# Copy the built frontend assets from the frontend-builder stage
COPY --from=frontend-builder /app/dist ./app/static

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
