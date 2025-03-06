# Use official Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app/backend

# Copy Python dependencies file
COPY requirements.txt .

# Install dependencies globally (NO venv)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Make the script executable
RUN chmod +x run_backend.sh

# Expose backend port
EXPOSE 8080

# Run the application
CMD ["bash", "./run_backend.sh"]
