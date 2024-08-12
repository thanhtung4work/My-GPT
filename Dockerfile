# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY ./train.py /app
COPY ./modules /app/modules
COPY ./requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (optional)
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Run the training script
CMD ["python", "train.py"]
