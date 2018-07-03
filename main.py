import tensorflow as tf
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random

# Import `tensorflow`
import tensorflow as tf


def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/hguan/project/machine-learning/garage/"
train_data_dir = os.path.join(ROOT_PATH, "training")
#test_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_dir)

images_array = np.array(images)
labels_array = np.array(labels)
print(len(set(labels_array)))

# Resize images
images32 = [transform.resize(image, (28, 28)) for image in images]
images32 = np.array(images32)

# Determine the (random) indexes of the images
traffic_signs = [1, 10, 9, 13]


# Resize images
images32 = [transform.resize(image, (28, 28)) for image in images]
images32 = np.array(images32)

images32 = rgb2gray(np.array(images32))
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)

plt.show()

print(images32.shape)