FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3.6 \
        python3.6-dev \
        python3-pip \
        python-setuptools \
        cmake \
        wget \
        curl \
        libsm6 \
        libxext6 \
        libxrender-dev

COPY requirements.gpu.txt /tmp

WORKDIR /tmp

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN python3.6 -m pip install -r requirements.gpu.txt

COPY . /All-In-One

WORKDIR /All-In-One

EXPOSE 50051

RUN cd models && chmod +x install.sh && ./install.sh

RUN python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. Service/service_spec/all_in_one.proto

CMD ["python3.6", "start_service.py"]