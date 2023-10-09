
# Deepfake Detection with CNNs

This repository contains code for detecting deepfakes in images using convolutional neural networks (CNNs). The code is written in Python and includes several Jupyter notebooks.

## Notebooks

- `dcgan_celeba.py`: This notebook contains code for training a DC-GAN on the CelebA dataset to generate fake images.

- `loading140k_dataset_and_pre_processing_.py`: This notebook contains code for loading and preprocessing the 140K dataset.

- `CNN-Networks`: This notebooks contains code for designing single-branch CNN networks and multi-branch CNN networks, as well as transfer learning methods using Inception-ResNet V2, DenseNet 121, and ResNet 50.
## datasets:
Celeb-a
The Celeb-a dataset is a large-scale face attributes dataset with more than 200K celebrity images. The images in this dataset are aligned and cropped to 178x218 pixels. We used this dataset to train a DC-GAN to generate fake images.

You can download the Celeb-a dataset from the following link:

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
140K
The 140K dataset is a large-scale deepfake detection dataset that contains 140K videos with real and fake faces. The videos in this dataset are of varying lengths and resolutions. We used this dataset to train and evaluate our deepfake detection models.

You can download the 140K dataset from the following link:

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
Note that you may need to create a Kaggle account and agree to the terms and conditions to download the dataset.

## Requirements

To run the code in this repository, you will need the following Python packages:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV

You can install these packages using pip:

```
pip install tensorflow keras numpy matplotlib opencv-python
```

## Usage

To use the code in this repository, simply clone the repository and run the Jupyter notebooks. You can modify the code to suit your needs and experiment with different models and datasets.

## Credits

This code was written by author of the following paper:

- [MB-CNN: A Multi-Branch
Convolutional Neural Network for
Deepfake Detection in Images]


If you use this code in your research, please cite the above papers.