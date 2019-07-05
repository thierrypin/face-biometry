# Face Biometry
## Complete pipeline for face biometry

- Face detection and alignment is performed by MTCNN, with a choice of implementation from Tensorflow or MXNet. It runs in real time on CPU, but there is a choice to run it on GPU.

- Feature extraction is performed by ArcFace over Mobilenet on MXNet. The vector length is 128.

This was possible by two github projects:


https://github.com/deepinsight/insightface/commit/be3f7b3e0e635b56d903d845640b048247c41c90

https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection/commit/b56065418b63a971fcf4f8f35d058513b0ce6cbf

