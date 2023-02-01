FROM tensorflow/tensorflow:1.12.0-gpu-py3

LABEL org.opencontainers.image.title="plantvillage-tf"
LABEL org.opencontainers.image.version="1.12"
LABEL org.opencontainers.image.description="PlantVillage customized version of Tensorflow image"

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y git
RUN pip install -U scikit-image
RUN pip install -U cython
RUN pip install 'git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'

WORKDIR /

