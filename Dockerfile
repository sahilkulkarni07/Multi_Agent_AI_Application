FROM python:3.10-slim

WORKDIR /app

# Install dependencies (replace with your actual requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python file(s)
COPY . .

# Run FastAPI App (adjust if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
