FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    WAREHOUSE_PATH=/app/data/warehouse.duckdb \
    TRACE_FILE=/app/traces/traces.jsonl

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir .

# Build the warehouse at image build time so containers start instantly and offline.
RUN python scripts/download_data.py && python scripts/build_warehouse.py \
    && rm data/Chinook_Sqlite.sqlite \
    && mkdir -p traces \
    && useradd --system --no-create-home app && chown -R app /app/traces
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "analytics_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
