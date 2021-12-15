import argparse
import glob
import random
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import lstsq
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

# Feel free to disable, QtAgg was freezing for me.
matplotlib.use("TKAgg", force=True)


class SDM:
    def __init__(self, train_img_paths, test_img_paths, savedir, name, orientations=9, pixels_per_cell=(8,8), max_iter=5):
        self.savedir = savedir
        self.name = name
        self.train_img_paths = train_img_paths
        self.test_img_paths = test_img_paths

        # HoG params
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell

        self.landmarks = None
        self.features = None
        self.pred_lm = None

        self.R = None
        self.max_iter = max_iter

    @staticmethod
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

    def get_image_features(self, img_paths):
        """Load each image into memory and extract a feature array.
        Returns in order of img_paths."""
        features = []
        for i, path in enumerate(img_paths):
            print(i, path)
            img = imread(path)
            img = resize(img, (128, 64))
            gray = rgb2gray(img)
            img_features = hog(gray, transform_sqrt=True, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell)
            features.append(img_features)

        features = np.stack(features)
        # Add bias column to features
        features = np.append(features, np.ones((features.shape[0], 1)), 1)

        return features

    def _populate(self):
        """Load training data, which takes a long time."""
        # Load training data
        self.landmarks = self.get_landmarks(self.train_img_paths)
        self.features = self.get_image_features(self.train_img_paths)

    def fit(self):
        # Reset regressor series
        self.R = []
        if not self.features:
            self._populate()

        # Pick an initialization- same for all images
        self.init_lm_single = self.landmarks.mean(axis=0)
        self.pred_lm = np.stack([self.init_lm_single.copy() for i in range(self.landmarks.shape[0])])

        # Train regressor matrices
        for i in range(self.max_iter):
            # Difference between initialization and ground truth landmarks
            delta_lm = self.landmarks - self.pred_lm
            # Fit gradient function based on features and known delta to ground truth
            R, *_ = lstsq(self.features, delta_lm)
            # Update training landmarks
            self.pred_lm = self.pred_lm + (self.features @ R)
            # Save regressor
            self.R.append(R)

        # Show results (training data)
        i = random.randint(0, len(self.train_img_paths)-1)
        plt.imshow(imread(self.train_img_paths[i]))
        plt.scatter(self.pred_lm.reshape(len(self.train_img_paths), -1, 2)[i, :, 0], self.pred_lm.reshape(len(self.train_img_paths), -1, 2)[i, :, 1], c="blue")
        plt.show()

        return self.pred_lm

    def predict(self, img_paths):
        features = self.get_image_features(img_paths)
        init_lm = np.stack([self.init_lm_single.copy() for i in range(len(self.test_img_paths))])
        pred_lm = init_lm
        # Apply regressors in sequence
        for R in self.R:
            pred_lm = pred_lm + (features @ R)

        # Show results
        i = random.randint(0, len(img_paths)-1)
        plt.imshow(imread(img_paths[i]))
        plt.scatter(pred_lm.reshape(len(img_paths), -1, 2)[i, :, 0], pred_lm.reshape(len(img_paths), -1, 2)[i, :, 1], c="blue")
        plt.show()

        return pred_lm

    def validate(self):
        return self.predict(self.test_img_paths)

    def save(self):
        os.makedirs(os.path.join(os.path.join(self.savedir, self.name)), exist_ok=False)
        np.savez(os.path.join(self.savedir, self.name, "matrices"), pred_lm=self.pred_lm, init_lm_single=self.init_lm_single, **{f"r{i}": r for i, r in enumerate(self.R)})
        open(os.path.join(self.savedir, self.name, "train_paths.txt"), "w").writelines([x + "\n" for x in self.train_img_paths])
        open(os.path.join(self.savedir, self.name, "test_paths.txt"), "w").writelines([x + "\n" for x in self.test_img_paths])

    def load(self):
        """Restore a saved model. This will overwrite all model params, except those passed to HoG."""
        loadpath = os.path.join(self.savedir, self.name)
        self.train_img_paths = [x.strip() for x in open(os.path.join(loadpath, "train_paths.txt"), "r").readlines()]
        self.test_img_paths = [x.strip() for x in open(os.path.join(loadpath, "test_paths.txt"), "r").readlines()]
        mats = np.load(os.path.join(loadpath, "matrices.npz"))
        self.pred_lm = mats["pred_lm"]
        self.init_lm_single = mats["init_lm_single"]
        self.R = [mats[k] for k in sorted([k for k in dict(mats).keys() if k[0] == "r"])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Descent Method (SDM) for aligning facial landmarks.")
    parser.add_argument("--trainset", type=str, help="Location of training data.", default="data/trainset")
    parser.add_argument("--testset", type=str, help="Location of test data.", default="data/testset")
    parser.add_argument("--trainlimit", type=int, help="Max number of training images to sample.", default=None)
    parser.add_argument("--testlimit", type=int, help="Max number of test images to sample.", default=None)
    parser.add_argument("--savedir", type=str, help="Directory in which to save model.", default="data/save")
    parser.add_argument("--modelname", type=str, help="Name of model, for saving.", default=str(time.time()).replace(".", "-"))
    parser.add_argument("--load", type=bool, help="Load a saved model using savedir and modelname.", default=False)

    args = parser.parse_args()

    train_img_paths = glob.glob(os.path.join(args.trainset, "*.jpg"))[:args.trainlimit]
    test_img_paths = glob.glob(os.path.join(args.testset, "*.jpg"))[:args.testlimit]
    sdm = SDM(train_img_paths=train_img_paths, test_img_paths=test_img_paths, name=args.modelname, savedir=args.savedir)

    if args.load:
        # This will overwrite all model params, except those passed to HoG.
        sdm.load()
    else:
        sdm.fit()

    sdm.validate()

    if not args.load:
        sdm.save()

    print("Done!")
