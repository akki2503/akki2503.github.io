# Runtime Model Optimization for Deep Learning Models - A Face Detection UseCase

Here's the table of contents:

1. TOC
{:toc}

## Problem Statement

[Deepface](https://github.com/serengil/deepface) is an open-source library for performing use cases like face detection, face recognition and building deep learning pipeline around these use cases.

The idea here is to use this library for performing face detection on videos. However the usual process of performing detection on each face is time taking and can lead to higher inference times on videos.

In this blog, we will explore and experiment with different techniques to optimize model runtime for tensorflow models.

## Setting up the basic infra and code

### Installing the required libraries
 > pip install tensorflow deepface python-opencv numpy 

 *Note*: To install tensorflow with gpu enabled follow the instructions on the official [tensorflow installation guide](https://www.tensorflow.org/install/pip).

### Setting up the basic code

```python
import cv2
import pathlib
import os
import time

from deepface.detectors import FaceDetector
from PIL import Image
import tensorflow as tf

# Define input and output paths
video_path = "/path/to/the/mp4/video"

# Initialize the face detector with retinaface backend
face_detector = FaceDetector.build_model(detector_backend = 'retinaface')

# Create a video object using cv2
video_object = cv2.VideoCapture(video_path)
read_frame_success = 1

# Read frames continuosly until the end of video
frame_counter = 0
start_time = time.time()
while read_frame_success:
    frame_counter+=1
    read_frame_success, image = video_object.read()
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    face_objs = FaceDetector.detect_faces(face_detector, "retinaface", image)
    print(f"Number of faces in frame {frame_counter} {len(face_objs)}")
    if frame_counter>=100:
        break
total_time = time.time() - start_time

print(f"Inference on 100 frames took {total_time} seconds")
```

## Evaluating Inference Time on cpu

In order to get the inference time for 100 frames on cpu add the following code the already existing basic code - 

```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

On an **Intel Corei7 Gen 10** CPU it takes - **144 seconds**

With this speed a 40 seconds 30 FPS video will take around 28-30 minutes. 

## Evaluating Inference Time on gpu

In order to get the inference time for 500 frames on cpu add the following code the already existing basic code - 

```python
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

On an **NVIDIA RTX 2060 6 GB** GPu it takes - **98 seconds**

With this speed a 40 seconds 30 FPS video will take around 3-4 minutes. 

## Evluating Inference Time on gpu with fp16 model

In order to get the inference time for 500 frames on cpu add the following code the already existing basic code - 

```python
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

On an **NVIDIA RTX 2060 6 GB** GPu it takes - **83 seconds**

With this speed a 40 seconds 30 FPS video will take around 3-4 minutes. 

## Conclusion about GPU and FP16 conversion

Although the conversion speeds up inference a lot, it is still far away from being used in production grade systems.

In order to make it faster, we can also try batch inferencing but for my usecases I need a batch size of 1 during inference so I will stop my experiments here for now.