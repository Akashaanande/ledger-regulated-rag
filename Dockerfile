FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/ledger/serve/app.py", "--server.address=0.0.0.0"]
