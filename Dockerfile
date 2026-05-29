FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY templates/ templates/
COPY justia_mvp.py .

EXPOSE 8000

CMD ["python", "-m", "src.main", "--web", "--host", "0.0.0.0", "--port", "8000"]
