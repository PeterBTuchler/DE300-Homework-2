FROM apache/spark-py:latest

USER root

WORKDIR /tmp

RUN apt update -y
RUN apt upgrade -y

RUN apt install -y python3.10-venv

RUN python3 -m venv demos

RUN source demos/bin/activate && \
pip3 install matplotlib venv-pack && \
pip3 install numpy venv-pack && \
pip3 install pandas venv-pack && \
demos/bin/venv-pack -o demos.tar.gz