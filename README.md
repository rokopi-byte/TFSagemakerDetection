# TFSagemakerDetection
Annotate, train on Sagemaker with Tensorflow Object Detection API, inference on Raspberry.
In this repo a complete pipeline for a custom dataset object detection is presented.
 
First, I would like to thank the following user for their work: [svpino](https://github.com/svpino/tensorflow-object-detection-sagemaker),  [douglasrizzo](https://github.com/douglasrizzo/detection_util_scripts)

Part of this repo is taken from them and slightly adapted and everything is put together.
We are going to use the Tensorflow Object Detection API v.1.14 on Tensorflow-GPU 1.15 for the training with a docker image on AWS Sagemaker.
Specifically we will use a pretained model for transfer learning on a custom class and inference on Raspberry PI 4.

You can choose to upload checkpoints and logs to S3 during training, in this way you can use your local tensorboard during training. Alternatively Sagemaker will keep logs and checkpoints in the EC2 instance and upload for you to S3 at the end of the training (together with the trained frozen model).  


Content of this repo:

* src_code: main code for the container to launch the training with Tensorflow Object Detection API.  
* Dockerfile: dockerfile to build the container with Tensorflow Object Detection API and Coco API installed. 
* Scripts for data preparation
* aws_credentials: fill the files inside this folder with credentials that have s3 permission. This is required only if you want to upload logs and checkpoints during training.

## Install Tensorflow Object Detection API

You can follow the official [repo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md) to install locally the Tensorflow Object Detection API. This [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/index.html) also cover most of things listed here, also explaining how to optionally install the COCO API for the metrics.

## Data preparation

First step is to prepare your data, more details [here](https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/using_your_own_dataset.md).

### Annotation
You can use [labelImg](https://github.com/tzutalin/labelImg) to annotate your images. You will obtain a single xml annotation for each image in your dataset. Suppose now you have a folder `img` with your images and a folder `annotations` with the annotations just generated.
Using the script `create_csv.py` transform all the xml files in a single csv file.

```bash
python3 create_csv.py xml annotations data.csv
```
### Label map generation
TensorFlow requires a label map file, which maps each of the used labels to an integer values. This file is specific to your dataset, unless you change the classes to detect you need to generate this only once.

```bash
python3 generate_label_map csv data.csv label_map.pbtxt
```

### Dataset Splitting
Split the dataset into train and validation. Arguments are `f` for the ratio (80-20 in this case) and `o` for an output folder where you will find the generated `data_train.csv` and `data_eval.csv`.

```bash
python3 generate_train_eval.py data.csv -f 0.80 -o outfolder
```

### TFRrecord generation

Generate [TFrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details) format for train and eval data. These files contain both the images and the annotations in a binary format. Arguments are the data csv, the label map, image folder and output name.


```bash
python3 generate_tfrecord.py data_train.csv label_map.pbtxt img train.record

python3 generate_tfrecord.py data_eval.csv label_map.pbtxt img eval.record
```

## Pretrained Model

Download the pretrained model from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

### Pipeline configuration

Everything you want to configure about the training job is inside `pipeline.config`. After downloading your pretrained model, you have to modify this file for your needs.
Main interest fields are:

* num_classes: put here number of classes you have to detect (COCO default it 90)

Under train_config set al the parameters for your training (batch_size, steps, optimizer..) and add this:
* fine_tune_checkpoint : "/opt/ml/input/data/checkpoint/model.ckpt"
* from_detection_checkpoint: true

Under tf_record_input_reader set:
* input_path: "/opt/ml/input/data/training/train.record"
* label_map_path: "/opt/ml/input/data/training/label_map.pbtxt"

The same for eval_input_reader:
* input_path: "/opt/ml/input/data/training/eval.record"
* label_map_path: "/opt/ml/input/data/training/label_map.pbtxt"

The path `/opt/ml/input/data/` is where Sagemaker will copy our input data in the machine instantiated for the training, and is fixed. We have then `training` and `checkpoint` which will we define as channels pointing to S3 locations.

## Amazon ECR

It's time to build our docker image (with the docker file provided) and push it to Amazon ECR registries  that will host our image. First create a repository on ECR, you will an URI like this:

    xxx.dkr.ecr.eu-west-1.amazonaws.com/<name>

Select the repository and click "View push commands" and just follow the instructions. Before that you have to configure the [AWS-Cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html). Our docker image with Tensorflow Object API is now ready to be deployed by Sagemaker.

## Amazon S3

Amazon S3 is the AWS storage service, and we use it to host all our input data and the output data (trained model, checkpoints.. )

Next step is to upload all the necessary to S3, the Amazon storage service. Create a bucket and name it `sagemaker-customname` ("sagemaker" in the name will give automatically permission to sagameker for read/write from the bucket). Inside the bucket create two folders: `training/` and `checkpoint` and upload the necessary files generated/downloaded before. Only if you intend to save chekpoints and logs during training, create an additional bucket `result` (the alternative way is that Sagemaker upload everything at the end of the training, in that case Sagemaker automatically create a folder with the training job name). 

This will be the structure in S3:

```
└── sagemaker-customname
     │
     ├─── training/
     │     ├── train.record
     │     ├── eval.record
     │     ├── pipeline.config
     │     └── label_map.pbtxt
     │
     ├─── checkpoint/
     │    ├── checkpoint
     │    ├── frozen_inference_graph.pb
     │    ├── model.ckpt.data-00000-of-00001
     │    ├── model.ckpt.meta
     │    └── model.ckpt.index
     │
     └─── result/
```

## Amazon Sagemaker 

Amazon SageMaker is a cloud machine-learning platform to create, train, and deploy machine-learning (ML) models in the cloud. We will use it only for the training here.
It's time create our training job: choose a name (a folder with this name will be created by Sagemaker in the S3 bucket at the end of the process, and will contain the training artifacts), create a new IAM role and select "Your own algorithm container in ECR" (your ECR URI defined above goes here). Then you can also define some regex, cloudwatch will use them to grab those values from the logs and plot them for you. It's an easy way to check in real time what's going on.
Example:
```
Metric Name:loss Regex:] loss = ([0-9\.]+)
Metric Name:step Regex:, step = ([0-9\.]+)
```

Choose your instance type, look [here](https://aws.amazon.com/it/ec2/instance-types/#Accelerated_Computing) for the features and [here](https://aws.amazon.com/it/sagemaker/pricing/) for the prices. For example a `ml.p2.xlarge` will give you a single K80 GPU for 1.2$/hr. You may need additional storage on the instance, you can add here. Don't forget to disable network isolation if you intend to upload logs and checkpoints during training from the EC2 instance to sagemaker.
This instance will be automatically closed after the training by Sagemaker, so you will only pay per second of training usage.
The following list of hyperparameters are supported by the training script embedded in your docker image:

### Hyperparameters
* num_steps: The number of training steps (defaults 100).

* quantize: Whether you want to also generate a TensorFlow Lite (TFLite) model that can be run on mobile devices. If not specified, this value defaults to False.

* image_size: The size that images will be resized to if we generating a TFLite model (both width and height will be set to this size.) This parameter is ignored if quantize is False. If not specified, this value defaults to 300.

* inference_type: The type of inference that will be used if we are generating a TFLite model. This should be one of QUANTIZED_UINT8 or FLOAT values. This parameter is ignored if quantize is False. If not specified, this value defaults to FLOAT.

* model_dir: (optional) You can specify a S3 bucket where tensorflow can send checkpoints and tfevents (logs for tensorboard) during training. In this way you can use tensorboard during training (see below) and not only at the end of training. In addition you can have checkpoints during training. This parameter is optional, if you omit it everything will be saved locally in the EC2 instance and copied back to S3 in a compressed archive. Example:

    `s3://sagemaker-customname/result`


### Input data configuration 
Create two channels under this section to allow SageMaker to expose the necessary resources to our docker image (select File for input mode):

* training: This channel exposes our data and configuration files to our docker image. Make sure to set the channel name property to "training". Example:

    `s3://sagemaker-customname/training`

* checkpoint: The second channel exposes our pre-trained network to our docker image. Make sure to set the channel name property to "checkpoint". Example:

   `s3://sagemaker-customname/checkpoint`

### Output data configuration
When the model finishes training, SageMaker can upload the results to our S3 bucket. Example:

`s3://sagemaker-customname`


You can now start the training job. During the training you can click on the training job and view under "monitor" your metrics plotted, clicking on "view logs" you will be redirected to Amazon Cloudwatch, where you can see all the output of the training algorithm logged.
At the end of the training, at the end of the page you have te link to directly download the model artifact, pushed back to our S3 bucket. You can point your local tensorboard to this folder to analyze in depth the training.

## Tensorboard

Install locally tensorboard and if you omitted `model_dir` hyperparameter simply download the model artifact from s3 at the end of the training, unpack it and launch:

```bash
tensorboard --logdir model
```

If you choose to send logs and checkpoints directly to s3 during training you can do this:

```bash
AWS_REGION=<your-region> AWS_LOG_LEVEL=3 tensorboard --logdir s3://sagemaker-customname/result/
```

Using tensorboard directly with s3 you will be charged with extra little cost. Amazon doesn't charge you for data transfer between services in the same region, so sending logs and checkpoint to S3 cost nothing in terms of data transfer (the same for the upload of the final artifact). Amazon anyway charge you for the operations (read/write/list..) and sending logs and checkpoints during training means many operations. In addition using tensorboard with s3 you pay for the number of operations on the bucket (tensorboard continuously polls the S3 filesystem to read logs), look [here](https://aws.amazon.com/s3/pricing/) under `data transfer` and for the operations on s3 in `requests and data retrievals`. 
You will anyway pay for the data transfer between S3 and your local machine.

During training and also during the use of tensorboard with S3 you can see errors like this (suppressed with AWS_LOG_LEVEL=3):

```
I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2020-11-23 11:41:02.502274: E tensorflow/core/platform/s3/aws_logging.cc:60] HTTP response code: 404
Exception name: 
Error message: No response body.
6 response headers:
connection : close
content-type : application/xml
date : Mon, 23 Nov 2020 10:41:01 GMT
server : AmazonS3
x-amz-id-2 : ...
x-amz-request-id : ...
2020-11-23 11:41:02.502364: W tensorflow/core/platform/s3/aws_logging.cc:57] If the signature check failed. This could be because of a time skew. Attempting to adjust the signer.
2020-11-23 11:41:02.502699: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2020-11-23 11:41:03.327409: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2020-11-23 11:41:03.491773: E tensorflow/core/platform/s3/aws_logging.cc:60] HTTP response code: 404
```

You can simply ignore them, this is the [reason](https://github.com/tensorflow/serving/issues/789#issuecomment-372817753).

## Inference on Raspberry

You should have downloaded the model artifacts, under model/graph you will find your frozen_inference_graph.pb which is the trained model. 
I use OpenCV for inference on Raspberry, in this case you need an additional configuration file as described [here](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API).
Be aware that OpenCV python wheels on raspberry are less updated, so check [here](https://www.piwheels.org/project/opencv-python/) if the version you need (OpenCV is constantly adding support for new network architecture) is available, otherwise the only option is to build OpenCV from source. You have basically 2 options, build directly on raspberry or as I recommend, cross-compile with [this](https://github.com/opencv/opencv/wiki/Intel%27s-Deep-Learning-Inference-Engine-backend#raspbian-buster) (remove the openvino things if you don't use it). 
Then it's just:

```python
import os
import time
import cv2 as cv

net = cv.dnn_DetectionModel('frozen_inference_graph.pb', 'cvgraph.pbtxt')

# The following is not necessary, change it if you use another backend like Intel Inference Engine of OpenVino
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

net.setInputSize(300, 300)
net.setInputSwapRB(True)

path = 'img'

images = [k for k in os.listdir(path)]

for image in images:
    frame = cv.imread(os.path.join(path, image))
    start_time = time.time()
    classes, confidences, boxes = net.detect(frame, confThreshold=0.60)
    elapsed_time = time.time() - start_time
    print('Elapsed time:', str(elapsed_time))
    if not len(boxes) == 0:
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            print(classId, confidence)
            cv.rectangle(frame, box, color=(0, 255, 0), thickness=2)
    cv.imshow('out', frame)
    cv.waitKey()

```