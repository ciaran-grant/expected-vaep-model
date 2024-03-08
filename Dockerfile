# Import Python Image
FROM python:3.11-slim

# Update package list
RUN apt-get update
# Install git
RUN apt-get install -y git

# Copy files in
WORKDIR /app
COPY . /app

# Install package
RUN pip install git+https://github.com/ciaran-grant/expected-vaep-model
RUN pip install -r requirements.txt

# Run app
CMD ["python", "app.py"] 