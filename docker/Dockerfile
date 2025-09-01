FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends cron jq supervisor && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x utils/*.sh

COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8050 8000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
