import glob
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from scipy.linalg import lstsq
from scipy.optimize import minimize
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

# Feel free to disable, QtAgg was freezing for me.
matplotlib.use("TKAgg", force=True)


def get_bboxes():
    # Just some ugly translations from the very nested MATLAB representation of
    # the bounding box information.
    bboxes1 = sio.loadmat('data\\Bounding Boxes\\bounding_boxes_helen_trainset.mat')['bounding_boxes']
    bboxes2 = sio.loadmat('data\\Bounding Boxes\\bounding_boxes_helen_testset.mat')['bounding_boxes']
    ret = {}
    for bb in bboxes1[0]:
        ret[bb[0][0][0][0]] = list(bb[0][0][1][0])
    for bb in bboxes2[0]:
        ret[bb[0][0][0][0]] = list(bb[0][0][1][0])
    return ret

def get_landmarks():
    """Extract landmark locations from .pts files in data folder."""
    train_landmarks = {}
    test_landmarks = {}

    for (dir, dataset) in [("trainset", train_landmarks), ("testset", test_landmarks)]:
        for lm_path in glob.glob(f"data/{dir}/*.pts"):
            lm = open(lm_path).readlines()

            # Parse and reformat into an array of pixel indices
            lm2 = []
            # Drop header and footer
            for line in lm[3:-1]:
                lm2.append(line.strip().split(" "))
            lm2 = np.array(lm2, dtype=float)

            # Store as dictionary keyed by image
            dataset[os.path.splitext(os.path.basename(lm_path))[0] + ".jpg"] = lm2

    return train_landmarks, test_landmarks

def get_image_features():
    """Load each images into memory and extract a feature array."""
    img_paths = glob.glob("data/trainset/*.jpg")
    fds = {} # TODO all images
    for i, path in enumerate(img_paths[:100]):
        print(i, path)
        img = imread(path)
        img = resize(img, (128, 64))
        gray = rgb2gray(img)
        fd = hog(gray, )
        fds[os.path.basename(path)] = fd

    return fds

fds = get_image_features()



# MAIN
# bboxes = get_bboxes()

# Read example image
imgpath = "232194_1.jpg"
img = imread(os.path.join("data/trainset/", imgpath))


# Landmarks
train_lm, test_lm = get_landmarks()
# x*
img_lm = train_lm[imgpath]
# x0 - average of all train landmarks
init_lm_single = np.stack(list(train_lm.values())).mean(axis=0)
init_lm = {k: init_lm_single.copy() for k in train_lm.keys()}
# ^x
delta_lm = img_lm - init_lm_single

# img_bbox = bboxes[imgpath]
gray = rgb2gray(img)
# Image
plt.imshow(img)
# Landmarks
plt.scatter(img_lm[:,0], img_lm[:,1], c="green")
plt.scatter(init_lm_single[:,0], init_lm_single[:,1], c="red")
# Covers img_lm, as it should
plt.scatter(init_lm_single[:,0] + delta_lm[:,0], init_lm_single[:,1] + delta_lm[:,1], c="blue")
plt.show()

# Get HoG features φ
fd, hog_image = hog(gray, visualize=True)
plt.imshow(hog_image)
plt.show()

# Define R0
R0 = np.zeros(shape=(delta_lm.shape[0], fd.shape[0]))

def descent_direction_solver(m_vector, x0, x_star, HoG_features, n_features):
    """Takes an array m_vector which is a concatenation of two matrices, R and B.
    R: (136, n_features)
    B: (n_features,)
    Therefore m_vector: (136 * n_features + n_features)

    Args:
    x0, set of initial landmarks.
    x_star, set of target landmarks.
    HoG_features, extracted using HoG (or other feature detector)
    n_features, from HoG

    This function will be passed to scipy.optimize.minimize (hence the
    single array parameter 'x').

    We need to solve for R and B in the expression:
    minimize:
        for all images:
            for all inits:
                norm(delta_x_star - R @ HoG_feature_vector - B)"""
    # x0 = args[0]
    # x_star = args[1]
    # HoG_features = args[2]
    # n_features = args[3]
    R = m_vector[:136 * n_features].reshape(136, n_features)
    B = m_vector[-136:]

    # SDM loss
    all_norms = []
    # for init_lm, target_lm in zip(x0, x_star):
    for path in HoG_features:
        delta_x_star = x_star[path] - x0[path]
        norm = np.linalg.norm(delta_x_star.flatten() - (R @ HoG_features[path]) - B)
        all_norms.append(norm)

    return sum(all_norms)

n_features = list(fds.values())[0].shape[0]

res = minimize(descent_direction_solver, x0=np.zeros(shape=(136 * n_features + 136), dtype=np.float16), args=(init_lm, train_lm, fds, n_features))



lstsq()


print("Done!")

# def test_minimize(m_vector, first_arg, second_arg):
#     return abs(first_arg - second_arg - m_vector)
# minimize(test_minimize, x0=np.array([100,]), args=(50, 20))

