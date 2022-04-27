FROM python:3.7-slim

ENV LANG=C.UTF-8

WORKDIR /usr/src/zoobot

RUN apt-get update && apt-get -y upgrade && \
  apt-get install --no-install-recommends -y \
  build-essential && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY README.md .
COPY setup.py .
RUN pip install -U .[pytorch]
# install the zoobot locally as a package
# COPY setup.py .
# RUN pip install -e .

# install package
COPY . .
