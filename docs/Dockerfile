FROM python:3.10

RUN apt-get update && apt-get install -y \
    git \
    make \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY docs/requirements.txt .
RUN pip install -r requirements.txt

#COPY setup.cfg .
#COPY pyproject.toml .
#COPY cebra/ .
