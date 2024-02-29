# Use a base image with Python installed
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install any additional dependencies required by Streamlit
RUN apt-get update && apt-get install -y sudo

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run Streamlit when the container starts
CMD ["streamlit", "run", "app.py"]
