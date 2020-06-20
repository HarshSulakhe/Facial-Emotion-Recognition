# Facial-Emotion-Recognition
Using Image Classification techniques to decipher the emotion of a person


## Dataset
[Link to Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)<br>

For easy usage of the fastai library, I chose to convert the csv file to ```.jpg``` images. If you wish to do the same, simply run ```python convert_csv_to_images.py``` after changing the appropriate paths within the script.

## Description
The notebook ```resnet.ipynb``` is a lazy implementation of transfer learning on ResNet34, ResNet50 and ResNet101. While the models appear to give a decent accuracy, it wasn't achieved without slight overfitting as can be seen in the loss plots of the 3 models.<br>

While looking for previous methods applied on the FER2013 dataset, I stumbled across [this repository](https://github.com/amilkh/cs230-fer). I attempted to replicate the transfer learning techniques (on ResNet50) as mentioned but failed to achieve the claimed accuracy. The possible reason for this could be the lack of usage of any external dataset than the one mentioned. Halfway into the training process, it was evident that the model was overfitting (in case of complete unfreezing) or had plateaued with a sub-par accuracy (in case of unfreezing only the last 5 layers). Please refer to ```resnet-custom.ipynb``` for the corresponding code.<br>

Overall though, this was a good learning process for me. Apart from learning how to freeze and unfreeze particular layers of a PyTorch model and customize layers of a pre-trained one,I figured out how to deal with training sample imbalances .Given the nature of the dataset, I also understood when and why to implement particular data augmentation techniques depending on the kind of data I'm working with.

## Dependencies
- PyTorch 1.4.0
- fastai 1.0.61
- Python 3.7.6
