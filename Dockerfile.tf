FROM tensorflow/tensorflow:2.8.0

# if you have a supported nvidia GPU and https://github.com/NVIDIA/nvidia-docker
# FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /usr/src/zoobot

# install dependencies but remove tensorflow as it's in the base image
COPY README.md .
COPY setup.py .
RUN pip install -U .[tensorflow]

# install package
COPY . .
