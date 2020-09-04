FROM nvidia/cuda:11.0-base
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
  build-essential \
  cmake \
  git \
  curl \
  vim \
  ca-certificates \
  libjpeg-dev \
  zip \
  unzip \
  libpng-dev

RUN apt-get -y install python3
RUN apt-get -y install python3-pip

COPY requirements.txt /ml-env/
WORKDIR /ml-env
RUN pip3 install -r requirements.txt
