FROM b.gcr.io/tensorflow/tensorflow:latest
MAINTAINER Ryan Houck <ryanchouck@gmail.com>

RUN sudo apt-get update

RUN sudo apt-get install -y python python-pip python-dev
RUN sudo apt-get install -y graphviz python-pygraphviz
RUN sudo apt-get install -y python-numpy python-scipy python-matplotlib

RUN pip install --upgrade pip

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN mkdir /project
WORKDIR /project