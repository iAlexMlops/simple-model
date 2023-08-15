FROM apache/spark-py:v3.4.0

USER 0

ENV TZ=Europe/Moscow
ENV PYTHONPATH=$PYTHONPATH:/app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt --no-cache-dir

WORKDIR /app

RUN chown -R 185 /app && chown -R 185 /usr/local/lib

USER 185

