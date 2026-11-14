FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8894
CMD ["python", "-c", "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=8894)"]
