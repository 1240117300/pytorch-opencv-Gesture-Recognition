# pytorch-opencv-Gesture-Recognition
The code in this experiment is based on python and uses pytorch to recognize the number represented by the gesture. First use opencv to open the camera to collect gesture grayscale images with a resolution of 64 * 48, then use pytorch to build a convolutional neural network to train the picture, get the training model, and finally call the model, use opencv to open the camera to collect real-time gesture photos , And finally display the detection result in the pycharm terminal.

The capture.py is used to collect collect gesture grayscale images with a resolution of 64 * 48
The test.py is used to train the model and detect the the picture of gesture realtime
