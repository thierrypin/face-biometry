# Face Biometry
## Complete pipeline for face biometry

This project contains no novelty in terms of computer vision techniques, but it integrates all the face biometry steps.

- Face detection and alignment is performed by MTCNN, with a choice of implementation from Tensorflow or MXNet. It runs in real time on CPU or GPU.

- Feature extraction is performed by ArcFace over Mobilenet on MXNet. The vector length is 128.


Other github projects made this possible:

- From [Facenet](https://github.com/davidsandberg/facenet), commit `51fc8cb7880f07c766dc1cc46a6f4f619dc5626c`, I got the Python/Tensorflow implementation of MTCNN.

- From [Insightface](https://github.com/deepinsight/insightface/), commit `be3f7b3e0e635b56d903d845640b048247c41c90`, I took the feature extraction model. For a compromise between speed and accuracy, I employ ArcFace + Mobilenet.

- From [MXNet MTCNN Face Detection](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection/), commit `b56065418b63a971fcf4f8f35d058513b0ce6cbf`, I took the Python/MXNet implementation of MTCNN.


## Requirements

This code requires the following packages:

- mxnet
- numpy
- opencv-python
- scikit-image
- scikit-learn
- tensorflow (optional)

Go to https://mxnet.apache.org/versions/master/install/index.html to find the adequate MXNet installation.

Tensorflow is optional, since detection defaults to MXNet.

## Running the demo

Create folder `facerec/data/rec/` and insert face pictures to register people. The images should contain exactly one face and their names should be in the format below.

`[person name]_[image number].[png|jpg|...]`.

Then run the demo script. Use `-h` to change parameters.

`python3 demo.py`


## The base classes

For face detection, use the `MTCNNDetector` class, and for feature extraction use the `Arcface` class. For the complete pipeline, including detection, encoding and classification, use the `FaceRecognition` interface.



