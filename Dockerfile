# ============================================================
# Base image
# ============================================================
FROM python:3.12-slim

# ---- System dependencies (for numpy, psycopg2, matplotlib, pymc) ----
    RUN apt-get update && apt-get install -y \
    libpq-dev gcc build-essential gfortran \
    libopenblas-dev liblapack-dev \
    libfreetype6-dev libpng-dev libjpeg-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Copy project ----
COPY . /app/

# ---- Python dependencies ----
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
    
# ---- Environment ----
ENV MODULE_FOLDER=/app/__module
ENV PYTHONUNBUFFERED=1
    

# ---- Default execution ----
WORKDIR /app/program
CMD ["python", "frequency.py", "0"]