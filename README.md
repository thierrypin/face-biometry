# Face Biometry
## Complete pipeline for face biometry

- Face detection and alignment is performed by MTCNN, with a choice of implementation from Tensorflow or MXNet. It runs in real time on CPU or GPU.

- Feature extraction is performed by ArcFace over Mobilenet on MXNet. The vector length is 128.


Two github projects made this possible:


https://github.com/deepinsight/insightface/commit/be3f7b3e0e635b56d903d845640b048247c41c90

https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection/commit/b56065418b63a971fcf4f8f35d058513b0ce6cbf


## Requirements

This code requires the following packages:

- mxnet
- numpy
- opencv-python
- scikit-image
- scikit-learn
- tensorflow

Go to https://mxnet.apache.org/versions/master/install/index.html to find the adequate MXNet installation.

Tensorflow is optional, since detection can also be done with MXNet (though it defaults to TF). Also, you may consider installing tensorflow-gpu if you have Cuda available.

## Running the demo

Create folder `facerec/data/rec/` and insert face pictures to register people. The images should contain exactly one face and their names should be in the format below.

`[person name]_[image number].[png|jpg|...]`.

Then run the demo script. Use `-h` to change parameters.

`python3 demo.py`


## The base classes

For face detection, use the `MTCNNDetector` class, and for feature extraction use the `Arcface` class. For the complete pipeline, including detection, encoding and classification, use the `FaceRecognition` interface.



