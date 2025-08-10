# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/tmp/hf_cache
ENV PYTHONUNBUFFERED=1


EXPOSE 7860


# Define the command to run the Streamlit app on port "7860" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
