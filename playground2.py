import glob
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from scipy.linalg import inv
from scipy.linalg import lstsq
from scipy.optimize import minimize
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

# Feel free to disable, QtAgg was freezing for me.
matplotlib.use("TKAgg", force=True)


def get_landmarks(img_paths):
    """Extract landmark locations from .pts files in data folder.
    Returns in order of img_paths."""
    landmarks = []

    for img_path in img_paths:
        # Read associated landmarks .pts file
        lm_path = os.path.splitext(img_path)[0] + ".pts"
        lm = open(lm_path).readlines()

        # Parse and reformat into an array of pixel indices
        lm2 = []
        # Drop header and footer
        for line in lm[3:-1]:
            lm2.append(line.strip().split(" "))
        lm2 = np.array(lm2, dtype=float).flatten()

        # Store in order
        landmarks.append(lm2)

    landmarks = np.stack(landmarks)

    return landmarks

def get_image_features(img_paths, limit=None):
    """Load each image into memory and extract a feature array.
    Returns in order of img_paths."""
    features = []
    for i, path in enumerate(img_paths[:limit]):
        print(i, path)
        img = imread(path)
        img = resize(img, (128, 64))
        gray = rgb2gray(img)
        img_features = hog(gray)
        features.append(img_features)

    features = np.stack(features)

    return features

def fit_descent_direction(init_lm, train_lm, features):
    """One iteration of SDM fit."""
    delta_lm = train_lm - init_lm
    features = np.append(features, np.ones((features.shape[0],1)),1 )
    # R = inv(features.T @ features) @ (features.T @ delta_lm)

    R, *_ = lstsq(features, delta_lm)
    b = R[-1, :]
    R = R[:-1]
    return R, b


# Main
img_paths = glob.glob("data/trainset/*jpg")
first_image = imread(img_paths[0])
train_lm = get_landmarks(img_paths)
# For debugging we use a limited number of images
features = get_image_features(img_paths, limit=100)
train_lm = train_lm[:features.shape[0]]

init_lm_single = train_lm.mean(axis=0)
init_lm = np.stack([init_lm_single for i in range(train_lm.shape[0])])

# Fit iteration 1
R0, b0 = fit_descent_direction(init_lm, train_lm, features)
# Update landmarks
iter_lm = init_lm + (features @ R0)

i = 31
plt.imshow(imread(img_paths[i]))
plt.scatter(init_lm.reshape(100, -1, 2)[i,:,0], init_lm.reshape(100, -1, 2)[i,:,1], c="green")
plt.scatter(iter_lm.reshape(100, -1, 2)[i,:,0], iter_lm.reshape(100, -1, 2)[i,:,1], c="blue")
plt.show()


print("Done!")

