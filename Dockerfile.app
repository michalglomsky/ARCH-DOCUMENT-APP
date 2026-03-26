# App server — browser UI + VLM proxy (CPU only)
FROM python:3.11-slim

COPY requirements/app.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY app/ /app/

# PDFs are expected at /data inside the container (mount your PDF folder there)
ENV PDF_DIR=/data
# Labels spreadsheet — set LABELS_XLSX env var if your xlsx has a different name
ENV LABELS_XLSX=/data/1-2026-DANE.xlsx
# VLM server address — overridden by docker-compose to use the service name
ENV VLM_URL=http://vlm-server:8081

EXPOSE 8000

CMD ["python", "server.py", "--port", "8000"]
