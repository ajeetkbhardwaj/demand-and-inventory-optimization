FROM python:3.10-slim

# Set up a working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U requests kaleido

# Copy the rest of the application files
COPY . .

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# Run Streamlit on port 7860
CMD ["streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]