The study is to develop an interpretable DL model capable of grading and localizing lumbar disc herniation (LDH).
The DL model comprises two components. Initially, we utilized Faster R-CNN with a ResNet 101 architecture to detect the ROI. The identified ROIs were segmented and then used for training the classification model.
Please cite our research if you use my code.
To set up an environment with TensorFlow 2.10, Python 3.9, CUDA 11.1, and cuDNN 8.2.
step_one for LDH_detection
step_two for LDH_classifier
