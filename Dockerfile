FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements.txt .
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then \
      PIP_CERT=/run/secrets/proxy_ca pip install --no-cache-dir -r requirements.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi
COPY . .
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
EXPOSE 10000
CMD ["python", "bot.py"]
