FROM python:3.10

WORKDIR /job

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "fairness_job.py"]
