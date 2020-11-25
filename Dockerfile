ARG ARCHITECTURE=1.15.0-gpu
FROM tensorflow/tensorflow:${ARCHITECTURE}-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget zip unzip git ca-certificates curl

# We need to install Protocol Buffers (Protobuf). Protobuf is Google's language and platform-neutral,  
# extensible mechanism for serializing structured data. To make sure you are using the most updated code,
# replace the linked release below with the latest version available on the Git repository.
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protoc-3.10.1-linux-x86_64.zip
RUN unzip protoc-3.10.1-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

# Let's add the folder that we are going to be using to install all of our machine learning-related code
# to the PATH. This is the folder used by SageMaker to find and run our code.
ENV PATH="/opt/ml/code:${PATH}"
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

RUN pip install --upgrade pip
RUN pip install cython
RUN pip install contextlib2
RUN pip install pillow
RUN pip install lxml
RUN pip install matplotlib
RUN pip install pycocotools

# Let's now download Tensorflow from the official Git repository and install Tensorflow Slim from
# its folder.
RUN git clone https://github.com/tensorflow/models/ tensorflow-models
RUN pip install -e tensorflow-models/research/slim

# We can now install the Object Detection API, also part of the Tensorflow repository. We are going to change
# the working directory for a minute so we can do this easily.
WORKDIR /opt/ml/code/tensorflow-models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN python setup.py build
RUN python setup.py install

# If you are interested in using COCO evaluation metrics, you can tun the following commands to add the
# necessary resources to your Tensorflow installation.
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /opt/ml/code/tensorflow-models/research/cocoapi/PythonAPI
RUN make 
RUN cp -r pycocotools /opt/ml/code/tensorflow-models/research/

# Let's put AWS credentials, you can skip this part if you don't want to put log and checkpoints to S3 during training. They will be
# uploaded anyway at the end of the training job, but you will not be able to use Tensorboard during training (only at the end)

WORKDIR /root
COPY /aws_credentials /root/.aws

# Let's put the working directory back to where it needs to be, copy all of our code, and update the PYTHONPATH
# to include the newly installed Tensorflow libraries.
WORKDIR /opt/ml/code
COPY /src_code /opt/ml/code
ENV PYTHONPATH=${PYTHONPATH}:tensorflow-models/research:tensorflow-models/research/slim:tensorflow-models/research/object_detection

RUN chmod +x train
