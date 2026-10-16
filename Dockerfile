# The leakcheck command-line audit in a small image.
#   docker build -t leakcheck .
#   docker run --rm -v "$PWD:/data" leakcheck scan --train /data/train.csv --test /data/test.csv --label label
FROM python:3.12-slim
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY leakcheck ./leakcheck
RUN pip install --no-cache-dir . && rm -rf /src
RUN useradd --create-home auditor
USER auditor
WORKDIR /data
ENTRYPOINT ["leakcheck"]
CMD ["--help"]
