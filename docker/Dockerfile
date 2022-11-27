FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Set system environment
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set working folder
WORKDIR /app

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install python
RUN apt-get update \
 && apt-get install -y --no-install-recommends git python3 python3-venv bc

# Install python dependencies
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip \
 && pip install wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt 

RUN rm -rf /var/lib/apt/lists/* /tmp/*
