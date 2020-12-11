FROM ubuntu:20.04

# Install Nginx.
RUN \
  apt-get update && \
  apt-get install -y vim && \
  apt-get install -y software-properties-common && \
  apt-get install -y curl


RUN \
  add-apt-repository -y ppa:deadsnakes/ppa && \
  apt-get install -y python3.8 python3.8-dev python3.8-venv python3-distutils python3-apt
  
RUN \
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
  python3.8 get-pip.py

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./static ./static
COPY . .

CMD ["/usr/bin/python3", "main.py"]
