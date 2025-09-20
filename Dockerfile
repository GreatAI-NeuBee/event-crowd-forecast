FROM python:3.12-slim

WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY deployment/ ./deployment/
COPY models/ ./models/
COPY src/ ./src/
COPY prediction/ ./prediction/
COPY data/ ./data

EXPOSE 8000

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
