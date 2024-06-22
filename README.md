**Fire Detection Classifier**

This project implements a fire detection classifier using a simple convolutional neural network (CNN) architecture. The CNN consists of 2 convolutional layers followed by max pooling, and 2 fully connected (linear) layers with a sigmoid activation function at the output.

**Dataset**
The dataset used for this project is sourced from Kaggle. You can find the dataset at Fire Dataset on Kaggle https://www.kaggle.com/datasets/phylake1337/fire-dataset.

**Data Distribution**
The dataset is split into training, validation, and test sets with the following label distributions:

**Label distribution:**

*Not-fire (class 0):* 172 samples, 34 samples, 37 samples (for training, validation, and testing)

*Fire (class 1):* 526 samples, 115 samples, 114 samples (for training, validation, and testing)



**The model achieved its best performance during training as follows:**

**Epoch [6/10]**

*Training Loss:* 0.1273, *Accuracy:* 95.56%

*Validation Loss:* 0.1332, *Accuracy:* 96.64%

**Overall Accuracy on Test Set:** 98.01%

*Precision:* 98.26%

*Recall:* 99.12%

*F1-score:* 98.69%

**Sample predictions:**
![photo_2024-06-22_21-17-29](https://github.com/MirasBaisbay/FireDetection/assets/107981665/e6a5304e-c4f1-4b32-9842-b00da868a57b)
