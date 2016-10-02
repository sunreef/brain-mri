import os
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import neural_net as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from voxel_grid import VoxelGrid

grid_size = 16

target_data = ""
with open('../data/set_train/targets.csv', 'r') as fo:
    target_data = fo.read()

numbers = target_data.split('\n')
numbers.pop(len(numbers) - 1)

vectorized_int = np.vectorize(int)
target_ages = vectorized_int(numbers)

feature_vectors = []
test_feature_vectors = []

for i in range(1, 279):
    save_path = "../data/features/grid_size_" + str(grid_size)
    try:
        vector = np.load(save_path + "/feature_vector_" + str(i) + ".npy")
    except:
        break
    feature_vectors.append(vector)

'''
# Not yet necessary.

for i in range(1, 138):
    save_path = "../data/test_features/grid_size_" + str(grid_size)
    try:
        vector = np.load(save_path + "/feature_vector_" + str(i) + ".npy")
    except:
        break
    test_feature_vectors.append(vector)
'''

if len(feature_vectors) < 278:
    for i in range(1, 279):
        img = nib.load("../data/set_train/train_" + str(i) + ".nii")
        grid = VoxelGrid(img, grid_size)

        save_path = "../data/features/grid_size_" + str(grid_size)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        feature_vectors.append(grid.get_feature_vector())
        np.save(save_path + "/feature_vector_" + str(i), feature_vectors[i - 1])
        print("Saved feature vector #" + str(i))

'''
# Not yet necessary.

if len(feature_vectors) < 138:
    for i in range(1, 138):
        img = nib.load("../data/set_test/test_" + str(i) + ".nii")
        grid = VoxelGrid(img, grid_size)

        save_path = "../data/test_features/grid_size_" + str(grid_size)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        feature_vectors.append(grid.get_feature_vector())
        np.save(save_path + "/feature_vector_" + str(i), feature_vectors[i - 1])
        print("Saved test feature vector #" + str(i))
'''

X = np.array(feature_vectors)
pca = PCA(n_components=8)
pca.fit(X)
reduced_X = pca.transform(X)

figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

reduced_X_x = reduced_X[:, 0:1]
reduced_X_y = reduced_X[:, 1:2]
reduced_X_z = reduced_X[:, 2:3]

color = np.zeros((278, 3))
max_age = np.max(target_ages)
min_age = np.min(target_ages)

for i in range(0, 278):
    color[i][0] = float(target_ages[i] - min_age) / (max_age - min_age)
    color[i][2] = 1.0 - float(target_ages[i] - min_age) / (max_age - min_age)

ax.scatter(reduced_X_x, reduced_X_y, reduced_X_z, c=color)
# plt.show()

# Scales data between 0 and 100.
reduced_scaled_X = MinMaxScaler((0, 100)).fit_transform(reduced_X)

# Neural Net
nn.try_params(reduced_scaled_X, target_ages)
