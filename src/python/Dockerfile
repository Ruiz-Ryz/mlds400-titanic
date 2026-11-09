FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY src/python/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src/python/main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
CMD ["--data-dir", "/app/data", "--output-dir", "/app/output"]
