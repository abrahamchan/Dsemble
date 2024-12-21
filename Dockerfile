# Base image
# Change base image below so that it matches CUDA version on host
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
FROM python

WORKDIR /home
RUN mkdir researcher
WORKDIR /home/researcher

ADD . /home/researcher/Dsemble
WORKDIR /home/researcher/Dsemble

ENV PYTHONPATH="/home/researcher/Dsemble/TFDM:${PYTHONPATH}"

RUN pip install -r requirements.txt

