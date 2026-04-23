FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY hmda_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY hmda_model.py hmda_api_features.py bootstrap_hmda_model.py group_fairness.py ./
COPY hmda_app/app.py ./
COPY hmda_app/templates ./templates
COPY hmda_app/static ./static
COPY hmda_app/entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV MODEL_PATH=/app/model_hmda.pkl
EXPOSE 5050

ENTRYPOINT ["./entrypoint.sh"]
