# Recoverable Privacy-Preserving Image Classification through Noise-like Adversarial Examples

This is the official code for the paper titled as "Recoverable Privacy-Preserving Image Classification through Noise-like Adversarial Examples".

## Preparation
Please download the datasets and the weight of RIC pre-trained on VGGFace2 or SVHN datasets from [GoogleDrive](https://drive.google.com/drive/folders/1YYEhhqvkO1VPrsbCx2rSTtY0PduxAd2P?usp=sharing).

For SVHN:

For VGGFace2:
Plead download the "VGGFace2_vggface2_train.tar" subset from [this url](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b), then name the extracted folder "train" and place it in the "data/vggface2" path.
## Train 

Take the training on the VGGFace2 dataset for an example:
1. Go to VGGFace2.py, set the variable stage='train'.
2. Run VGGFace2.py

## Test

Take the training on the VGGFace2 dataset for an example,
1. Go to VGGFace2.py, set the variable stage='test'.
2. Run VGGFace2.py



