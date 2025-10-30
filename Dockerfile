# Stage 1: Define the base environment
# Use a Python 3.9 slim image for a smaller, faster container
FROM python:3.12-slim-bullseye

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create and set the working directory inside the container
WORKDIR $APP_HOME

# 1. Copy the core definition files needed for installation
# Copy these first to leverage Docker's build cache
COPY requirements.txt .
COPY setup.py .

# 2. Install Python dependencies and your local package
# Use --no-cache-dir for clean, lean installation
RUN pip install --no-cache-dir -r requirements.txt
# Install your local EAP package based on setup.py
RUN pip install .

# 3. Copy the rest of the application code and artifacts
# CRITICAL: This copies the EAP package, app.py, templates, and the artifact files!
COPY EAP/ EAP/
COPY app.py .
COPY templates/ templates/
COPY artifacts/ artifacts/
COPY schema.yaml .

# 4. Expose the port (Flask default)
EXPOSE 5000


CMD ["python3", "app.py"]