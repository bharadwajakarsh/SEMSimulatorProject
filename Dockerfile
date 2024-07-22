# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /app/home/jovyan/SEMSimulatorProject


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME SEM_DEFAULT_ENV

# Run app.py when the container launches
CMD ["python3", "display_and_analysis_functions.py"]