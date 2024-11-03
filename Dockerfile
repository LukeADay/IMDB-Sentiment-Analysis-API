# Start with a base image that has Python installed
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the model, tokenizer, and application code
COPY sentiment_model.keras tokenizer.pkl app.py /app/

# Expose the FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
