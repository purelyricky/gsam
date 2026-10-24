FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-lock.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt

COPY . .
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
