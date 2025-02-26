# Use official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy Python dependencies file
COPY requirements.txt .

# Install dependencies globally (NO venv)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Make the script executable
RUN chmod +x run_project.sh

# Expose backend and frontend ports
EXPOSE 8080

# Run the application
CMD ["bash", "./run_project.sh"]

